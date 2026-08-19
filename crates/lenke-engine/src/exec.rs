//! Execution: pull a batch up through the plan, then materialize the projection.
//!
//! Expression evaluation is columnar — `eval` produces a `Col` over the whole
//! batch, reading typed storage columns in bulk where it can. It calls the value
//! contract for every comparison and equality; it never restates those rules.
//! This is the lineage-FREE strategy; the lineage-preserving strategy for the
//! same operators lands with the operators (path/tags) that need it.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::batch::{Batch, Col, Lineage};
use crate::ir::{Agg, AggFn, CombineOp, CompareOp, Dir, Expr, PathMode, Plan, TxKind};
use crate::store::{Column, Store};
use crate::value::{self, Value};

/// Mulberry32 PRNG — the fully-specified generator `sample()` uses with a FIXED seed,
/// byte-identical to lenke-core (and the TS engine): same seed + same draw order ⇒ the
/// same seeded shuffle on every engine.
struct Mulberry32 {
    s: u32,
}
impl Mulberry32 {
    fn new(seed: u32) -> Self {
        Self { s: seed }
    }
    fn next_f64(&mut self) -> f64 {
        self.s = self.s.wrapping_add(0x6d2b_79f5);
        let mut t = (self.s ^ (self.s >> 15)).wrapping_mul(1u32 | self.s);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61u32 | t));
        f64::from(t ^ (t >> 14)) / 4_294_967_296.0
    }
}

/// A fast, dependency-free hasher for the engine's INTERNAL grouping, distinct,
/// and join maps. The default `HashMap` hasher (SipHash) is DoS-resistant, which
/// these maps — built and dropped inside one operator over trusted, already-
/// materialized keys — do not need; FNV-1a is several times faster on the short
/// byte/integer keys grouping produces. It never escapes the executor, so hash
/// quality only affects speed, never results.
pub(crate) mod fnv {
    use std::collections::{HashMap, HashSet};
    use std::hash::{BuildHasherDefault, Hasher};

    pub type Map<K, V> = HashMap<K, V, BuildHasherDefault<Fnv>>;
    pub type Set<K> = HashSet<K, BuildHasherDefault<Fnv>>;

    pub struct Fnv(u64);

    impl Default for Fnv {
        fn default() -> Self {
            Self(0xcbf2_9ce4_8422_2325) // FNV-1a 64-bit offset basis
        }
    }

    impl Hasher for Fnv {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, bytes: &[u8]) {
            let mut h = self.0;
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
            }
            self.0 = h;
        }
    }
}
use fnv::{Map as FnvMap, Set as FnvSet};

/// A materialized result: column names and rows of values. `Value` intentionally
/// has no `PartialEq` (f64/NaN policy lives in the value contract, not a derive),
/// so compare results through `value::equals`/`cmp_total`, not `==`.
/// Result cells in a FLAT row-major buffer (`data[i*ncols + j]`) — one allocation
/// for the whole result instead of a `Vec` per row. The nested `Vec<Vec<Value>>`
/// layout measured ~4x slower to build (a malloc per row), and this matches core's
/// `RowSet`. It still indexes and iterates like the old nested layout —
/// `flat[i]` / `flat[i][j]` yield a row slice / cell, `flat.len()` the row count,
/// `flat.iter()` (and `&flat`) yield `&[Value]` rows — so read sites are unchanged;
/// only construction goes through [`Flat::from_rows`] or the direct push in `run`.
#[derive(Debug, Clone, Default)]
pub struct Flat {
    data: Vec<Value>,
    ncols: usize,
}

impl Flat {
    fn with_capacity(nrows: usize, ncols: usize) -> Self {
        Self {
            data: Vec::with_capacity(nrows.saturating_mul(ncols)),
            ncols,
        }
    }
    /// The number of rows (`data.len() / ncols`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.ncols).unwrap_or(0)
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// Iterate rows as `&[Value]` slices — the drop-in for the old `Vec::iter`.
    pub fn iter(&self) -> std::slice::Chunks<'_, Value> {
        self.data.chunks(self.ncols.max(1))
    }
    /// Build from the nested layout — the construction path for tests and callers
    /// that already hold a `Vec<Vec<Value>>`.
    #[must_use]
    pub fn from_rows(rows: Vec<Vec<Value>>) -> Self {
        let ncols = rows.first().map_or(0, Vec::len);
        let mut data = Vec::with_capacity(rows.len().saturating_mul(ncols));
        for r in rows {
            data.extend(r);
        }
        Self { data, ncols }
    }
}

impl std::ops::Index<usize> for Flat {
    type Output = [Value];
    fn index(&self, i: usize) -> &[Value] {
        let c = self.ncols.max(1);
        &self.data[i * c..i * c + c]
    }
}

impl<'a> IntoIterator for &'a Flat {
    type Item = &'a [Value];
    type IntoIter = std::slice::Chunks<'a, Value>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug)]
pub struct Rows {
    pub names: Vec<String>,
    pub rows: Flat,
}

/// Run `plan` over `store`, returning materialized rows. Output column names come
/// from the outermost naming operator (`Project` or `Aggregate`, seen through
/// `Distinct`/`OrderPage`); a plan with none surfaces slot 0 under a single
/// implicit column so partial plans stay runnable in tests.
#[must_use]
pub fn run(plan: &Plan, store: &Store) -> Rows {
    try_run(plan, store).expect("read plan evaluation faulted")
}

/// The fallible core of [`run`]: an expression can fault at runtime (a failed
/// `CAST` throws `E_INVALID_VALUE`), so the read pipeline threads a `Result`. A
/// plan that never evaluates a fallible expression cannot error, which is why
/// [`run`] can wrap this with `.expect` — the panic path is unreachable for such
/// plans, and callers that may run user CASTs use `try_run` (or `execute`).
/// Does the plan contain a variable-length / repeat / shortest-path operator? Those run a
/// recursive DFS whose depth is the traversal depth, so an unbounded quantifier on a deep
/// graph can recurse far enough to overflow the default 8 MB stack (a stack overflow
/// aborts the process — `catch_unwind` cannot recover it). Such a plan runs on a
/// large-stack thread instead. Everything else keeps the cheap direct path.
// Only the non-wasm `on_big_stack` consults this; wasm runs traversals inline.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn plan_has_varlen(plan: &Plan) -> bool {
    match plan {
        // `VarLength` now runs the ITERATIVE `varlen_walk` (O(1) call stack — see
        // `deep_varlen_walk_runs_on_a_tiny_stack`), so it never *commits* the big stack.
        // We still take the reservation: it is virtual (≈0 RSS) and protects the whole
        // read pipeline around the walk, which does use call-stack proportional to plan
        // shape. The others below still recurse on data and genuinely need it.
        Plan::VarLength { .. }
        | Plan::RepeatGroup { .. }
        | Plan::NestedGroup { .. }
        | Plan::ShortestPath { .. }
        | Plan::ShortestPathEnum { .. } => true,
        Plan::Scan { .. }
        | Plan::NodeSeed { .. }
        | Plan::EdgeScan
        | Plan::EdgeSeed { .. }
        | Plan::Row
        | Plan::IndexSeek { .. }
        | Plan::RangeSeek { .. }
        | Plan::Insert { .. }
        | Plan::InsertReturn { .. }
        | Plan::Merge { .. }
        | Plan::AddEdge { .. }
        | Plan::CallProcedure { .. } => false,
        Plan::Filter { input, .. }
        | Plan::Project { input, .. }
        | Plan::Aggregate { input, .. }
        | Plan::Expand { input, .. }
        | Plan::EdgeVertex { input, .. }
        | Plan::Distinct { input }
        | Plan::DistinctBy { input, .. }
        | Plan::OrderPage { input, .. }
        | Plan::Sample { input, .. }
        | Plan::Enumerate { input, .. }
        | Plan::GroupToMap { input }
        | Plan::SortLocal { input, .. }
        | Plan::Unwind { input, .. }
        | Plan::Update { input, .. }
        | Plan::CallInline { input, .. } => plan_has_varlen(input),
        Plan::Join { left, right, .. } | Plan::Union { left, right, .. } => {
            plan_has_varlen(left) || plan_has_varlen(right)
        }
        // Rare/other operators (tree, subgraph, interval/optional expand, algo wrappers):
        // conservatively use the big stack — they may wrap a traversal, and the cost of a
        // reserved-but-untouched stack is nil.
        _ => true,
    }
}

/// Run `f` on a large (virtual) stack when `plan` contains a deep-recursion traversal, so
/// an unbounded quantifier's recursive DFS can't overflow the default 8 MB stack; a simple
/// plan runs `f` directly. Reserving a big stack costs nothing until the recursion uses
/// it. Used by every entry that may drive a traversal — `try_run` and the Gremlin-JSON
/// sinks, which call `pull` off the main path.
#[cfg(not(target_arch = "wasm32"))]
fn on_big_stack<T: Send>(plan: &Plan, f: impl FnOnce() -> T + Send) -> T {
    if !plan_has_varlen(plan) {
        return f();
    }
    const BIG_STACK: usize = 1 << 30; // 1 GiB
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(BIG_STACK)
            .spawn_scoped(s, f)
            .expect("spawn traversal thread")
            .join()
            .expect("traversal thread panicked")
    })
}

/// wasm has no threads, so a deep traversal runs inline on the module's own stack.
/// If an unbounded quantifier overflows it, raise the wasm stack size at link time
/// (`-C link-arg=-zstack-size=…`) rather than spawning — there is nothing to spawn.
#[cfg(target_arch = "wasm32")]
fn on_big_stack<T>(_plan: &Plan, f: impl FnOnce() -> T) -> T {
    f()
}

pub fn try_run(plan: &Plan, store: &Store) -> Result<Rows, String> {
    on_big_stack(plan, || try_run_inner(plan, store))
}

fn try_run_inner(plan: &Plan, store: &Store) -> Result<Rows, String> {
    // Lineage is plan-global: if anything reads the path, the whole plan tracks
    // it (Scan seeds, Expand extends); otherwise no operator builds a sidecar and
    // the query pays nothing for lineage.
    let track = needs_lineage(plan);
    let batch = match pull_top_output_streamed(plan, store, track)? {
        Some(b) => b,
        None => pull(plan, store, track)?,
    };
    let n = batch.rows();
    // CONSUME the batch to build the result — the final render is the LAST use of these
    // columns, so a materialized Str/Num/Bool/Gen cell MOVES into its `Value` instead of
    // being cloned out of a column we then drop. The old build re-cloned every cell (a
    // second full materialization on top of the projection's Col::Str), doubling the
    // per-row Arc-clone work a string projection pays.
    let names = output_names(plan).unwrap_or_else(|| vec!["_".to_string()]);
    let ncols = names.len();
    let slots = batch.slots;
    let data = if ncols == 1 {
        // Single output column (values/label/id/count, or a one-item RETURN): its cells
        // ARE the rows — move them straight into a row-major buffer, no transpose.
        let col = slots
            .into_iter()
            .next()
            .unwrap_or_else(|| Col::Gen(vec![Value::Null; n]));
        col_into_values(col, store)
    } else {
        // Multi-column: a row-major buffer filled column-by-column (each column consumed).
        let mut data = vec![Value::Null; n * ncols];
        for (c, col) in slots.into_iter().enumerate() {
            if c >= ncols {
                break; // defensive: never write past the declared columns
            }
            render_col_into(col, store, &mut data, c, ncols);
        }
        data
    };
    Ok(Rows {
        names,
        rows: Flat { data, ncols },
    })
}

/// Move a single output column's cells into a fresh row-major `Vec<Value>`, CONSUMING
/// the column: a `Str`/`Num`/`Bool` cell moves into its `Value` (no clone), a `Gen`
/// column IS already `Vec<Value>` (returned as-is, zero work), and a node/edge frontier
/// renders to its element value via the store. Matches `render_cell` exactly, minus the
/// clone.
fn col_into_values(col: Col, store: &Store) -> Vec<Value> {
    match col {
        Col::Str(data) => data.into_iter().map(Value::Str).collect(),
        Col::Num(data) => data.into_iter().map(Value::Num).collect(),
        Col::Bool(data) => data.into_iter().map(Value::Bool).collect(),
        Col::Gen(data) => data,
        Col::Nodes(ids) => render_nodes(store, &ids),
        Col::Edges(eids) => eids
            .into_iter()
            .map(|e| {
                if e == u32::MAX {
                    Value::Null
                } else {
                    edge_result_value(store, e)
                }
            })
            .collect(),
    }
}

/// The multi-column twin of [`col_into_values`]: move column `col`'s cells into the
/// row-major `out` buffer at column `c` (stride `ncols`), consuming the column.
fn render_col_into(col: Col, store: &Store, out: &mut [Value], c: usize, ncols: usize) {
    match col {
        Col::Str(data) => {
            for (i, s) in data.into_iter().enumerate() {
                out[i * ncols + c] = Value::Str(s);
            }
        }
        Col::Num(data) => {
            for (i, x) in data.into_iter().enumerate() {
                out[i * ncols + c] = Value::Num(x);
            }
        }
        Col::Bool(data) => {
            for (i, b) in data.into_iter().enumerate() {
                out[i * ncols + c] = Value::Bool(b);
            }
        }
        Col::Gen(data) => {
            for (i, v) in data.into_iter().enumerate() {
                out[i * ncols + c] = v;
            }
        }
        Col::Nodes(ids) => {
            for (i, v) in render_nodes(store, &ids).into_iter().enumerate() {
                out[i * ncols + c] = v;
            }
        }
        Col::Edges(eids) => {
            for (i, e) in eids.into_iter().enumerate() {
                out[i * ncols + c] = if e == u32::MAX {
                    Value::Null
                } else {
                    edge_result_value(store, e)
                };
            }
        }
    }
}

/// Run a plan that MAY write, against a mutable store. A write plan (`Insert`)
/// mutates the store and returns no rows; any other plan is a pure read and is
/// dispatched to [`run`] over a shared borrow. This is the entry point for
/// statements that can mutate; read-only callers can keep using [`run`]. Returns
/// `Err` when a write violates a constraint (the write is rolled back); reads and
/// successful writes are `Ok` (a write's result is the empty row set).
/// Run an `Insert`'s writes (nodes then edges) inside a transaction, enforcing
/// unique + required constraints on every touched label and rolling the whole
/// statement back on the first violation. Returns the ids of the created nodes,
/// in creation order (index i is the node declared at position i). Shared by
/// A per-statement atomic scope. A GQL write wraps its mutations so a constraint
/// violation rolls back exactly that statement. If a transaction is ALREADY open
/// (an explicit `transaction()`), the scope is a SAVEPOINT within it — the statement
/// undoes to its own mark on failure without abandoning the caller's transaction, and
/// its changes commit with the caller's. Standalone, it is an implicit single-statement
/// `begin`/`commit`. This is what lets the same write run both bare and nested (the
/// unconditional `begin` used to panic "nested transactions are not supported").
#[derive(Clone, Copy)]
enum StmtScope {
    /// No transaction was open: this scope owns an implicit begin/commit.
    Implicit,
    /// A transaction was already open: a savepoint at this undo-log mark.
    Nested(usize),
}

fn stmt_begin(store: &mut Store) -> StmtScope {
    if store.in_transaction() {
        StmtScope::Nested(store.savepoint())
    } else {
        store.begin();
        StmtScope::Implicit
    }
}

/// End a statement scope on SUCCESS: an implicit scope commits; a nested scope leaves
/// its changes in the enclosing transaction (committed when the caller commits).
fn stmt_commit(store: &mut Store, scope: StmtScope) {
    if matches!(scope, StmtScope::Implicit) {
        store.commit();
    }
}

/// End a statement scope on FAILURE: an implicit scope rolls the whole thing back; a
/// nested scope rolls back only to its savepoint, keeping the caller's transaction open.
fn stmt_rollback(store: &mut Store, scope: StmtScope) {
    match scope {
        StmtScope::Implicit => store.rollback(),
        StmtScope::Nested(mark) => store.rollback_to(mark),
    }
}

/// Run a write statement's DEFERRED declared-constraint checks (unique / required /
/// type / cardinality / validators / invariants) — but ONLY when the statement is
/// standalone (`Implicit` scope, its own auto-commit transaction). Inside an explicit
/// transaction (`Nested`) the checks are DEFERRED to that transaction's COMMIT
/// ([`commit_with_deferred_checks`]), so a later statement can complete a state that is
/// temporarily invalid mid-transaction — the SQL DEFERRABLE-constraint model core uses.
/// Immediate faults (a string-`id` collision, a syntax error) are NOT deferred; they
/// still roll the one statement back at the write site.
fn check_deferred_if_standalone(store: &Store, scope: StmtScope) -> Result<(), String> {
    if matches!(scope, StmtScope::Implicit) {
        store
            .run_deferred_checks()
            .and_then(|()| enforce_expr_constraints(store))
    } else {
        Ok(())
    }
}

/// Commit an explicit transaction: run the deferred declared-constraint checks against
/// the fully-staged graph, then commit — or roll the WHOLE transaction back on the first
/// violation. Shared by the GQL `COMMIT` keyword and the host `transaction()` commit.
pub(crate) fn commit_with_deferred_checks(store: &mut Store) -> Result<(), String> {
    if let Err(e) = store
        .run_deferred_checks()
        .and_then(|()| enforce_expr_constraints(store))
    {
        store.rollback();
        return Err(e);
    }
    store.commit();
    Ok(())
}

/// Execute an ISO GQL transaction-control command against the store's transaction
/// frame, returning an empty result (no rows/columns), like a write-only query. ISO
/// semantics are enforced HERE (matching core's `run_tx_control`), every violation
/// carrying the `E_INVALID_GRAPH_OP` wire code: `START TRANSACTION` while one is
/// active → error (no nesting); `COMMIT` / `ROLLBACK` with no active transaction →
/// error. The transaction persists across `lnk_query` calls (the store IS the
/// session), so a `START` here stays open for later statements until a `COMMIT` /
/// `ROLLBACK`. `READ ONLY` is recorded on the store (cleared on commit/rollback) for
/// a later write to consult.
pub(crate) fn run_tx_control(
    store: &mut Store,
    kind: TxKind,
    read_only: bool,
) -> Result<Rows, String> {
    match kind {
        TxKind::Start => {
            if store.in_transaction() {
                return Err(
                    "E_INVALID_GRAPH_OP: START TRANSACTION: a transaction is already active".into(),
                );
            }
            store.begin();
            store.set_tx_read_only(read_only);
        }
        TxKind::Commit => {
            if !store.in_transaction() {
                return Err("E_INVALID_GRAPH_OP: COMMIT: no active transaction".into());
            }
            // Run the DEFERRED declared-constraint checks against the fully-staged
            // graph; a violation rolls the whole transaction back (read-only mode
            // clears either way).
            let checked = commit_with_deferred_checks(store);
            store.set_tx_read_only(false);
            checked?;
        }
        TxKind::Rollback => {
            if !store.in_transaction() {
                return Err("E_INVALID_GRAPH_OP: ROLLBACK: no active transaction".into());
            }
            store.rollback();
            store.set_tx_read_only(false);
        }
    }
    Ok(empty_rows())
}

/// Reject a write statement issued inside a `READ ONLY` transaction, before it
/// applies. Called by the FFI write path (a read is always allowed). Matches core's
/// `enforce_read_only`.
pub(crate) fn enforce_read_only(store: &Store) -> Result<(), String> {
    if store.tx_read_only() {
        return Err(
            "E_INVALID_GRAPH_OP: write statement rejected: the active transaction is READ ONLY"
                .into(),
        );
    }
    Ok(())
}

/// The outcome of [`run_query`]: either rows already materialized (a transaction
/// -control command or a write), or a read whose optimized plan is handed back so
/// the caller can take its language-specific STREAMING JSON path (the perf-critical
/// route we must not collapse into a materialized batch).
pub enum Executed {
    /// A `TxControl` result or a write's returned rows — already produced.
    Rows(Rows),
    /// A read: the optimized plan, for the caller to stream.
    Read(Plan),
}

/// The single query entry point for a parsed plan against a MUTABLE store. In one
/// place — so every embedder (the C ABI, a future in-process host) gets identical
/// semantics — it dispatches ISO transaction control (`START TRANSACTION`/`COMMIT`/
/// `ROLLBACK`), enforces `READ ONLY`, and splits writes from reads. This mirrors
/// core, whose eval layer (not its ABI) owns transaction dispatch; previously the
/// engine duplicated this decision at each FFI language arm.
///
/// A `TxControl` command yields no rows and is neither optimized nor planned. Every
/// other plan is optimized here; a write is rejected under a READ ONLY transaction,
/// then executed; a read is returned as [`Executed::Read`] for the caller to stream.
pub fn run_query(plan: Plan, store: &mut Store) -> Result<Executed, String> {
    if let Plan::TxControl { kind, read_only } = plan {
        return run_tx_control(store, kind, read_only).map(Executed::Rows);
    }
    let plan = crate::opt::optimize_indexed(plan, store);
    if is_write(&plan) {
        enforce_read_only(store)?;
        execute(&plan, store).map(Executed::Rows)
    } else {
        Ok(Executed::Read(plan))
    }
}

/// The `FAULT_ID_DUP` message, matching core: a string `id` is an element's unique
/// external identity. The `E_UNIQUE:` prefix maps to `E_CONSTRAINT_VIOLATION`.
const ID_DUP_ERR: &str = "E_UNIQUE: an element with this id already exists — a string `id` \
     property is the element's unique identity; use _MERGE to upsert, or a fresh id";

/// The `FAULT_ID_IMMUTABLE` message, matching core: a string `id` is an element's
/// fixed identity, so `SET x.id = …` is rejected (`E_INVALID_GRAPH_OP`).
const ID_IMMUTABLE_ERR: &str = "E_INVALID_GRAPH_OP: cannot SET `id`: a string `id` is the \
     element's identity and is fixed at creation — insert a new element with the new id instead";

/// Whether node `n`'s `id` PROPERTY is its identity — a string equal to its external
/// id (as set by `INSERT (:P {id: 'x'})`). Such an id is fixed at creation, so a
/// `SET n.id` is rejected; a numeric / absent / divergent `id` is an ordinary,
/// SET-able property. Stateless, matching core's `vertex_id_is_identity`.
fn node_id_is_identity(store: &Store, n: u32) -> bool {
    if let Value::Str(s) = store.prop(n, "id") {
        store
            .node_ext_id(n)
            .is_some_and(|ext| ext.as_ref() == s.as_ref())
    } else {
        false
    }
}

/// The edge analogue of [`node_id_is_identity`] — matching core's `edge_id_is_identity`.
fn edge_id_is_identity(store: &Store, e: u32) -> bool {
    if let Value::Str(s) = store.edge_prop(e, "id") {
        store
            .edge_ext_id(e)
            .is_some_and(|ext| ext.as_ref() == s.as_ref())
    } else {
        false
    }
}

/// Create a node for an INSERT, honoring a string `id` property as the element's
/// EXTERNAL identity (like core's `insert_vertex_with_id`): a string `id` becomes the
/// node's external id — unique across the graph, a duplicate is a constraint violation
/// — AND is still stored as an ordinary property (`RETURN n.id` works). A non-string
/// or absent `id` mints a synthetic external id. (Numeric `id` stays a plain property.)
fn insert_node_with_identity(
    store: &mut Store,
    labels: &[&str],
    props: &[(&str, Value)],
) -> Result<u32, String> {
    if let Some((_, Value::Str(id))) = props.iter().find(|(k, _)| *k == "id") {
        if store.node_by_ext(id).is_some() {
            return Err(ID_DUP_ERR.into());
        }
        let ext: std::sync::Arc<str> = std::sync::Arc::from(id.as_ref());
        return Ok(store.add_node_with_id(&ext, labels, props));
    }
    Ok(store.add_node(labels, props))
}

/// Create an edge for an INSERT, honoring a string `id` property as its external
/// identity (like core's `insert_edge_with_id`): a string `id` becomes the edge's
/// external id — unique among edges, a duplicate is a constraint violation — and is
/// still stored as a property. A non-string / absent `id` mints a synthetic edge id.
/// Sets the edge's properties in either case.
fn insert_edge_with_identity(
    store: &mut Store,
    from: u32,
    to: u32,
    etype: &str,
    props: &[(String, Value)],
) -> Result<u32, String> {
    let eid = if let Some((_, Value::Str(id))) = props.iter().find(|(k, _)| k == "id") {
        if store.edge_by_ext(id).is_some() {
            return Err(ID_DUP_ERR.into());
        }
        let ext: std::sync::Arc<str> = std::sync::Arc::from(id.as_ref());
        store.add_edge_with_id(&ext, from, to, etype)
    } else {
        store.add_edge(from, to, etype)
    };
    for (k, v) in props {
        store.set_edge_prop(eid, k, v.clone());
    }
    Ok(eid)
}

/// `Plan::Insert` and `Plan::InsertReturn`.
fn run_insert(
    store: &mut Store,
    nodes: &[crate::ir::InsertNode],
    edges: &[crate::ir::InsertEdge],
) -> Result<Vec<u32>, String> {
    // In a transaction so a constraint violation (or a duplicate string `id`) rolls
    // the whole INSERT back rather than leaving a partial write.
    let scope = stmt_begin(store);
    let result = (|| -> Result<Vec<u32>, String> {
        let mut ids = Vec::with_capacity(nodes.len());
        for spec in nodes {
            let labels: Vec<&str> = spec.labels.iter().map(String::as_str).collect();
            let props: Vec<(&str, Value)> = spec
                .props
                .iter()
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect();
            ids.push(insert_node_with_identity(store, &labels, &props)?);
        }
        for e in edges {
            insert_edge_with_identity(store, ids[e.from], ids[e.to], &e.etype, &e.props)?;
        }
        // Enforce the declared constraints this INSERT could have violated (unique,
        // required, type, cardinality, validators, invariants) — NOW if standalone,
        // else DEFERRED to the enclosing transaction's COMMIT.
        check_deferred_if_standalone(store, scope)?;
        Ok(ids)
    })();
    match result {
        Ok(ids) => {
            stmt_commit(store, scope);
            Ok(ids)
        }
        Err(e) => {
            stmt_rollback(store, scope);
            Err(e)
        }
    }
}

/// `Plan::InsertFrom` — a row-driven INSERT (`FOR … INSERT`, `MATCH … INSERT`).
/// Evaluate the templates' property expressions over the input rows (read phase,
/// immutable borrow), then create each row's nodes/edges (write phase). The whole
/// statement is ONE atomic scope — a constraint violation on any row rolls back
/// EVERY row (so `FOR x IN [1,2] INSERT (:U {id:'dup'})` under a unique constraint
/// leaves zero rows), matching `run_insert`'s per-statement atomicity.
fn run_insert_from(
    store: &mut Store,
    input: &Plan,
    nodes: &[crate::ir::InsertNodeExpr],
    edges: &[crate::ir::InsertEdgeExpr],
) -> Result<(), String> {
    // Read phase: pull the rows and materialize every template property to an OWNED
    // per-row value vector, so the immutable borrow ends before the write phase.
    let (node_props, edge_props, nrows) = {
        let batch = pull(input, store, needs_lineage(input))?;
        let nrows = batch.rows();
        let eval_props = |props: &[(String, Expr)]| -> Result<Vec<(String, Vec<Value>)>, String> {
            props
                .iter()
                .map(|(k, e)| {
                    let col = eval(e, store, &batch)?;
                    Ok((k.clone(), (0..nrows).map(|i| col.value_at(i)).collect()))
                })
                .collect()
        };
        let node_props: Vec<_> = nodes
            .iter()
            .map(|n| eval_props(&n.props))
            .collect::<Result<_, _>>()?;
        let edge_props: Vec<_> = edges
            .iter()
            .map(|e| eval_props(&e.props))
            .collect::<Result<_, _>>()?;
        (node_props, edge_props, nrows)
    };
    // Write phase: create every row's nodes/edges under one per-statement scope. A
    // string `id` property is the node's external identity (unique) — a duplicate
    // (across rows or with an existing node) rolls back EVERY row.
    let scope = stmt_begin(store);
    let result = (|| -> Result<(), String> {
        for i in 0..nrows {
            let mut row_ids = Vec::with_capacity(nodes.len());
            for (t, n) in nodes.iter().enumerate() {
                let labels: Vec<&str> = n.labels.iter().map(String::as_str).collect();
                let props: Vec<(&str, Value)> = node_props[t]
                    .iter()
                    .map(|(k, vals)| (k.as_str(), vals[i].clone()))
                    .collect();
                row_ids.push(insert_node_with_identity(store, &labels, &props)?);
            }
            for (t, e) in edges.iter().enumerate() {
                let row_props: Vec<(String, Value)> = edge_props[t]
                    .iter()
                    .map(|(k, vals)| (k.clone(), vals[i].clone()))
                    .collect();
                insert_edge_with_identity(
                    store,
                    row_ids[e.from],
                    row_ids[e.to],
                    &e.etype,
                    &row_props,
                )?;
            }
        }
        check_deferred_if_standalone(store, scope)
    })();
    if let Err(err) = result {
        stmt_rollback(store, scope);
        return Err(err);
    }
    stmt_commit(store, scope);
    Ok(())
}

/// Whether `plan`'s root is a write (INSERT / SET / REMOVE / DELETE / _MERGE /
/// addE). A write must run through [`execute`] (mutable store); a read goes through
/// the immutable [`try_run`] path. The C ABI's `lnk_query` routes on this.
// Only the `capi` ffi layer consults it; without that feature it is unused.
#[cfg_attr(not(feature = "capi"), allow(dead_code))]
pub(crate) fn is_write(plan: &Plan) -> bool {
    matches!(
        plan,
        Plan::Insert { .. }
            | Plan::InsertFrom { .. }
            | Plan::InsertReturn { .. }
            | Plan::Update { .. }
            | Plan::UpdateReturn { .. }
            | Plan::Merge { .. }
            | Plan::AddEdge { .. }
    )
}

pub fn execute(plan: &Plan, store: &mut Store) -> Result<Rows, String> {
    match plan {
        Plan::Insert { nodes, edges } => {
            run_insert(store, nodes, edges)?;
            Ok(empty_rows())
        }
        Plan::InsertFrom {
            input,
            nodes,
            edges,
        } => {
            run_insert_from(store, input, nodes, edges)?;
            Ok(empty_rows())
        }
        Plan::InsertReturn { nodes, edges, tail } => {
            // First write-then-return path: run the INSERT, then bind each created
            // node into the slot equal to its creation index and project the tail.
            let ids = run_insert(store, nodes, edges)?;
            // A one-row seed: slot i carries the id of the i-th created node, so the
            // tail's `Expr::Prop{slot}` reads the node just created at that index.
            let seed = Batch::of(ids.iter().map(|&id| Col::Nodes(vec![id])).collect());
            // The tail is restricted (by the parser + this guard) to pure
            // projections; `pull_body` covers Row/Project/Filter, not the read
            // pipeline's grouping/paging operators.
            let store_ref: &Store = store;
            let batch = pull_body(tail, store_ref, &seed)?;
            Ok(rows_from_batch(tail, &batch, store_ref))
        }
        Plan::Update { input, ops } => {
            run_update(store, input, ops, needs_lineage(input))?;
            Ok(empty_rows())
        }
        Plan::UpdateReturn { input, ops, tail } => {
            // Read-after-write: run `input`, apply `ops`, then read `tail` over the
            // SAME frontier against the mutated store (write-then-return, the twin of
            // InsertReturn). `run_update` returns the frontier it wrote so the tail
            // reads the just-written values without re-scanning.
            let track = needs_lineage(input) || needs_lineage(tail);
            let frontier = run_update(store, input, ops, track)?;
            let store_ref: &Store = store;
            let batch = pull_body(tail, store_ref, &frontier)?;
            Ok(rows_from_batch(tail, &batch, store_ref))
        }
        Plan::Merge {
            label,
            props,
            on_create,
            on_update,
        } => execute_merge(store, label, props, on_create, on_update),
        Plan::AddEdge {
            from,
            to,
            etype,
            props,
        } => {
            let nc = u32::try_from(store.node_count()).unwrap_or(u32::MAX);
            if *from >= nc || *to >= nc || !store.is_alive(*from) || !store.is_alive(*to) {
                return Err(format!(
                    "addE: endpoint out of range or deleted ({from} -> {to})"
                ));
            }
            let eid = store.add_edge(*from, *to, etype);
            for (k, v) in props {
                store.set_edge_prop(eid, k, v.clone());
            }
            Ok(empty_rows())
        }
        _ => try_run(plan, store),
    }
}

/// Render a tail projection's output `Batch` to `Rows` — the shared tail of the
/// write-then-return paths (`InsertReturn` / `UpdateReturn`). Uses the tail's output
/// names when it is a projection, else a single `_` column over slot 0.
fn rows_from_batch(tail: &Plan, batch: &Batch, store: &Store) -> Rows {
    let n = batch.rows();
    match output_names(tail) {
        Some(names) => {
            let ncols = names.len();
            let mut rows = Flat::with_capacity(n, ncols);
            for i in 0..n {
                for c in &batch.slots {
                    rows.data.push(render_cell(c, i, store));
                }
            }
            Rows { names, rows }
        }
        None => {
            let slot0 = batch.slot(0);
            let mut rows = Flat::with_capacity(n, 1);
            for i in 0..n {
                rows.data.push(render_cell(slot0, i, store));
            }
            Rows {
                names: vec!["_".to_string()],
                rows,
            }
        }
    }
}

/// The concrete write a matched row expands to, computed while the store is still
/// immutably borrowed (the read phase) so the write phase can mutate freely.
enum Applied {
    Set(u32, String, Value),
    Remove(u32, String),
    DeleteNode(u32, bool), // (node, detach)
    DeleteEdge(u32),       // eid
    SetEdge(u32, String, Value),
    RemoveEdge(u32, String),
}

/// Run a `MATCH … SET/REMOVE/DELETE` update: pull the frontier, apply the ops in a
/// statement transaction (with the deferred-constraint recheck), and RETURN the
/// frontier batch it operated on — so `UpdateReturn` can read the just-written values
/// over the same bindings. `Plan::Update` discards the returned batch.
fn run_update(
    store: &mut Store,
    input: &Plan,
    ops: &[crate::ir::SetOp],
    track: bool,
) -> Result<Batch, String> {
    // Read phase: run the match and compute every write into OWNED data — the pulled
    // frontier batch owns its columns (no store borrow), so it survives the mutation
    // and seeds an `UpdateReturn` tail.
    let frontier = pull(input, store, track)?;
    let mut applied: Vec<Applied> = Vec::new();
    {
        let batch = &frontier;
        for op in ops {
            match op {
                crate::ir::SetOp::Set { slot, key, value } => {
                    let vals = eval(value, store, batch)?;
                    match batch.slot(*slot) {
                        Col::Nodes(ids) => {
                            for (i, &id) in ids.iter().enumerate() {
                                // A string `id` is the element's fixed identity —
                                // re-keying it would break element_id / round-trip.
                                if key == "id" && node_id_is_identity(store, id) {
                                    return Err(ID_IMMUTABLE_ERR.into());
                                }
                                applied.push(Applied::Set(id, key.clone(), vals.value_at(i)));
                            }
                        }
                        Col::Edges(eids) => {
                            for (i, &e) in eids.iter().enumerate() {
                                if key == "id" && edge_id_is_identity(store, e) {
                                    return Err(ID_IMMUTABLE_ERR.into());
                                }
                                applied.push(Applied::SetEdge(e, key.clone(), vals.value_at(i)));
                            }
                        }
                        _ => {}
                    }
                }
                crate::ir::SetOp::Remove { slot, key } => match batch.slot(*slot) {
                    Col::Nodes(ids) => {
                        for &id in ids {
                            applied.push(Applied::Remove(id, key.clone()));
                        }
                    }
                    Col::Edges(eids) => {
                        for &e in eids {
                            applied.push(Applied::RemoveEdge(e, key.clone()));
                        }
                    }
                    _ => {}
                },
                crate::ir::SetOp::Delete { slot, detach } => match batch.slot(*slot) {
                    Col::Nodes(ids) => {
                        for &id in ids {
                            applied.push(Applied::DeleteNode(id, *detach));
                        }
                    }
                    Col::Edges(eids) => {
                        for &e in eids {
                            applied.push(Applied::DeleteEdge(e));
                        }
                    }
                    _ => {}
                },
            }
        }
    }
    // Write phase, as a TRANSACTION so a constraint violation rolls the whole
    // statement back — matching INSERT/_MERGE. Previously SET/REMOVE applied
    // with no recheck, so `SET u.email = <existing>` silently violated a
    // unique constraint and `REMOVE u.email` a required one.
    let scope = stmt_begin(store);
    // Pass 1: property writes and EDGE deletes. Node deletes are deferred to
    // pass 2 so an edge deleted here (`DELETE r, a, b`) leaves its endpoints
    // relationship-free before the non-DETACH node-delete check runs.
    let mut node_deletes: Vec<(u32, bool)> = Vec::new();
    for a in applied {
        match a {
            Applied::Set(node, key, value) => store.set_prop(node, &key, value),
            Applied::Remove(node, key) => store.remove_prop(node, &key),
            Applied::SetEdge(eid, key, value) => store.set_edge_prop(eid, &key, value),
            Applied::RemoveEdge(eid, key) => store.remove_edge_prop(eid, &key),
            Applied::DeleteEdge(eid) => {
                if let Some((u, v)) = store.edge_endpoints(eid) {
                    store.delete_edge(u, v, eid);
                }
            }
            Applied::DeleteNode(node, detach) => node_deletes.push((node, detach)),
        }
    }
    // Pass 2: node deletes. A non-DETACH delete of a node that still has
    // relationships is an error (Cypher/core semantics); DETACH deletes the
    // incident edges too (delete_node cascades). A node matched by several
    // rows is deleted once (skip if already gone).
    for (node, detach) in node_deletes {
        if !store.is_alive(node) {
            continue;
        }
        if !detach && (!store.out(node).is_empty() || !store.inc(node).is_empty()) {
            stmt_rollback(store, scope);
            return Err(
                "E_INVALID_GRAPH_OP: cannot DELETE a node that still has relationships; \
                         use DETACH DELETE"
                    .into(),
            );
        }
        store.delete_node(node);
    }
    // Declared-constraint checks: NOW if standalone, else deferred to the
    // enclosing transaction's COMMIT. Roll back on the first violation.
    if let Err(e) = check_deferred_if_standalone(store, scope) {
        stmt_rollback(store, scope);
        return Err(e);
    }
    stmt_commit(store, scope);
    Ok(frontier)
}

/// Execute a `_MERGE`: infer the key from a unique constraint, find the existing
/// node by its key values, and take the create or update path. Runs in a
/// transaction so a constraint violation (or a no-applicable-constraint error)
/// leaves the store untouched.
fn execute_merge(
    store: &mut Store,
    label: &str,
    props: &[(String, Value)],
    on_create: &[(String, Expr)],
    on_update: &crate::ir::MergeUpdate,
) -> Result<Rows, String> {
    use crate::ir::MergeUpdate;
    let scope = stmt_begin(store);
    let have: Vec<String> = props.iter().map(|(k, _)| k.clone()).collect();
    let key_keys = match store.infer_merge_key(label, &have) {
        Ok(k) => k,
        Err(e) => {
            stmt_rollback(store, scope);
            return Err(e);
        }
    };
    // The pattern's key-tuple bytes, and a finder that matches an existing node.
    let want = key_bytes(&key_keys, |k| pattern_value(props, k));
    let found = store
        .nodes_with_label(label)
        .iter()
        .copied()
        .find(|&id| key_bytes(&key_keys, |k| store.prop(id, k)) == want);

    match found {
        Some(id) => match on_update {
            MergeUpdate::Nothing => {}
            MergeUpdate::Clobber => {
                // Set every non-key payload property to the pattern's value.
                for (k, v) in props {
                    if !key_keys.contains(k) {
                        store.set_prop(id, k, v.clone());
                    }
                }
            }
            MergeUpdate::Set { assigns, filter } => {
                let batch = Batch::of(vec![Col::Nodes(vec![id])]);
                // Evaluate the gate and every assignment BEFORE mutating; a fault
                // (e.g. a failed CAST) rolls the whole MERGE back rather than
                // leaving the begun transaction open.
                let gate = match filter.as_ref().map(|f| eval(f, store, &batch)).transpose() {
                    Ok(g) => g.is_none_or(|c| matches!(c.value_at(0), Value::Bool(true))),
                    Err(e) => {
                        stmt_rollback(store, scope);
                        return Err(e);
                    }
                };
                if gate {
                    let writes: Result<Vec<(String, Value)>, String> = assigns
                        .iter()
                        .map(|(k, e)| Ok((k.clone(), eval(e, store, &batch)?.value_at(0))))
                        .collect();
                    match writes {
                        Ok(writes) => {
                            for (k, v) in writes {
                                store.set_prop(id, &k, v);
                            }
                        }
                        Err(e) => {
                            stmt_rollback(store, scope);
                            return Err(e);
                        }
                    }
                }
            }
        },
        None => {
            let props_ref: Vec<(&str, Value)> =
                props.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            let id = store.add_node(&[label], &props_ref);
            let batch = Batch::of(vec![Col::Nodes(vec![id])]);
            let writes: Result<Vec<(String, Value)>, String> = on_create
                .iter()
                .map(|(k, e)| Ok((k.clone(), eval(e, store, &batch)?.value_at(0))))
                .collect();
            match writes {
                Ok(writes) => {
                    for (k, v) in writes {
                        store.set_prop(id, &k, v);
                    }
                }
                Err(e) => {
                    stmt_rollback(store, scope);
                    return Err(e);
                }
            }
        }
    }

    if let Err(e) = check_deferred_if_standalone(store, scope) {
        stmt_rollback(store, scope);
        return Err(e);
    }
    stmt_commit(store, scope);
    Ok(empty_rows())
}

// -------------------------------------------------- validators / invariants ---
//
// These two constraint kinds need the query evaluator, so — unlike the pure-store
// constraints in `run_deferred_checks` — they are declared and enforced here.
// A validator is checked by the composed query `MATCH (var:target) WHERE NOT
// (pred) …`: three-valued `WHERE` keeps only rows where `pred` is definitely
// FALSE, so a non-empty result is exactly an SQL-`CHECK` violation (null/true
// pass). An invariant is its own whole-graph query; a boolean-`false` cell fails.

/// Apply one schema-DDL op. Validator/invariant ops are declared here (parse,
/// run the declaration-time check over existing data, then store); every other op
/// delegates to the pure-store [`crate::schema_op::apply`]. This is the single
/// schema entry point the C ABI calls.
pub fn apply_schema_op(store: &mut Store, json: &str) -> Result<(), crate::schema_op::SchemaError> {
    use crate::schema_op::SchemaError;
    let parsed = crate::ndjson::parse_json(json).map_err(SchemaError::BadRequest)?;
    let crate::ndjson::Json::Obj(fields) = &parsed else {
        return Err(SchemaError::BadRequest(
            "schema op must be a JSON object".into(),
        ));
    };
    let op = crate::ndjson::field(fields, "op")
        .and_then(|j| crate::ndjson::json_string(j).ok())
        .ok_or_else(|| SchemaError::BadRequest("schema op needs a string `op`".into()))?;
    match op.as_str() {
        "validator" => declare_validator_op(store, fields),
        "invariant" => declare_invariant_op(store, fields),
        _ => crate::schema_op::apply(store, json),
    }
}

/// Read a required string field, as a `BadRequest` on any shape error.
fn schema_str(
    fields: &[(String, crate::ndjson::Json)],
    key: &str,
) -> Result<String, crate::schema_op::SchemaError> {
    use crate::schema_op::SchemaError;
    let j = crate::ndjson::req(fields, key).map_err(SchemaError::BadRequest)?;
    crate::ndjson::json_string(j)
        .map_err(|e| SchemaError::BadRequest(format!("field `{key}`: {e}")))
}

fn declare_validator_op(
    store: &mut Store,
    fields: &[(String, crate::ndjson::Json)],
) -> Result<(), crate::schema_op::SchemaError> {
    let target = schema_str(fields, "label")?;
    let var = schema_str(fields, "var")?;
    let pred = schema_str(fields, "predicate")?;
    declare_validator(store, &target, &var, &pred)
}

/// Declare a validator: compose + parse the vertex/edge check queries, verify the
/// current data conforms, then store the rule. Shared by the schema-op path and
/// the binary-snapshot reload.
pub(crate) fn declare_validator(
    store: &mut Store,
    target: &str,
    var: &str,
    pred: &str,
) -> Result<(), crate::schema_op::SchemaError> {
    use crate::schema_op::SchemaError;
    let vq = format!("MATCH ({var}:{target}) WHERE NOT ({pred}) RETURN {var} LIMIT 1");
    let eq = format!("MATCH ()-[{var}:{target}]->() WHERE NOT ({pred}) RETURN {var} LIMIT 1");
    let vplan = crate::gql::parse(&vq).map_err(SchemaError::Syntax)?;
    let eplan = crate::gql::parse(&eq).map_err(SchemaError::Syntax)?;
    for plan in [&vplan, &eplan] {
        if !try_run(plan, store)
            .map_err(SchemaError::Rejected)?
            .rows
            .is_empty()
        {
            return Err(SchemaError::Rejected(
                "existing data already violates the validator being declared".into(),
            ));
        }
    }
    store.declare_validator(target, var, pred, vec![vplan, eplan]);
    Ok(())
}

fn declare_invariant_op(
    store: &mut Store,
    fields: &[(String, crate::ndjson::Json)],
) -> Result<(), crate::schema_op::SchemaError> {
    let name = schema_str(fields, "name")?;
    let query = schema_str(fields, "query")?;
    declare_invariant(store, &name, &query)
}

/// Declare an invariant: parse the query, verify the current data holds, then
/// store it. Shared by the schema-op path and the binary-snapshot reload.
pub(crate) fn declare_invariant(
    store: &mut Store,
    name: &str,
    query: &str,
) -> Result<(), crate::schema_op::SchemaError> {
    use crate::schema_op::SchemaError;
    let plan = crate::gql::parse(query).map_err(SchemaError::Syntax)?;
    if rows_have_false(&try_run(&plan, store).map_err(SchemaError::Rejected)?) {
        return Err(SchemaError::Rejected(format!(
            "existing data already violates the invariant '{name}'"
        )));
    }
    store.declare_invariant(name, query, plan);
    Ok(())
}

/// Run every validator + invariant after a write statement; the caller rolls the
/// statement back on `Err`. A no-op when none are declared.
pub(crate) fn enforce_expr_constraints(store: &Store) -> Result<(), String> {
    for plan in store.validator_check_plans() {
        if !try_run(plan, store)?.rows.is_empty() {
            return Err("E_VALIDATOR: a validator predicate was violated".to_string());
        }
    }
    for (name, plan) in store.invariant_plans() {
        if rows_have_false(&try_run(plan, store)?) {
            return Err(format!("E_INVARIANT: invariant '{name}' violated"));
        }
    }
    Ok(())
}

/// Whether any cell in a result is boolean `false` — the invariant-violation test.
fn rows_have_false(rows: &Rows) -> bool {
    rows.rows
        .iter()
        .any(|r| r.iter().any(|c| matches!(c, Value::Bool(false))))
}

/// The grouping-key bytes of `keys`, reading each key's value via `get`.
fn key_bytes(keys: &[String], mut get: impl FnMut(&str) -> Value) -> Vec<u8> {
    let mut buf = Vec::new();
    for k in keys {
        value::group_key_into(&get(k), &mut buf);
    }
    buf
}

/// A pattern property's value by key (NULL if the pattern does not name it).
fn pattern_value(props: &[(String, Value)], key: &str) -> Value {
    props
        .iter()
        .find(|(k, _)| k == key)
        .map_or(Value::Null, |(_, v)| v.clone())
}

/// Render a batch cell (slot `col`, row `i`) to a result `Value`. A NODE frontier
/// slot renders as core's element MAP `{id, labels, properties}` (not its bare id),
/// so `RETURN n` / `RETURN *` match core byte-for-byte. Everything else materializes
/// as its plain value. (Edge frontier rendering — `{id, from, to, labels,
/// properties}` — needs an eid→endpoints accessor and is a separate step.)
fn render_cell(col: &Col, i: usize, store: &Store) -> Value {
    match col {
        // `u32::MAX` is the OPTIONAL-MATCH null sentinel → NULL, not an element map.
        Col::Nodes(ids) if ids[i] == u32::MAX => Value::Null,
        Col::Edges(eids) if eids[i] == u32::MAX => Value::Null,
        Col::Nodes(ids) => node_result_value(store, ids[i]),
        Col::Edges(eids) => edge_result_value(store, eids[i]),
        _ => col.value_at(i),
    }
}

/// The `id` field of a bare-VERTEX element map (`{id, labels, properties}`), or `None`.
fn vertex_map_ext_id(v: &Value) -> Option<&str> {
    let Value::Map(pairs) = v else { return None };
    let keys: std::collections::BTreeSet<&str> = pairs
        .iter()
        .filter_map(|(k, _)| match k {
            Value::Str(s) => Some(s.as_ref()),
            _ => None,
        })
        .collect();
    if keys.len() != pairs.len() || keys != ["id", "labels", "properties"].into_iter().collect() {
        return None;
    }
    pairs.iter().find_map(|(k, val)| match (k, val) {
        (Value::Str(k), Value::Str(id)) if k.as_ref() == "id" => Some(id.as_ref()),
        _ => None,
    })
}

/// The `id` field of a bare-EDGE element map (`{id, from, to, labels, properties}`).
fn edge_map_ext_id(v: &Value) -> Option<&str> {
    let Value::Map(pairs) = v else { return None };
    let keys: std::collections::BTreeSet<&str> = pairs
        .iter()
        .filter_map(|(k, _)| match k {
            Value::Str(s) => Some(s.as_ref()),
            _ => None,
        })
        .collect();
    if keys.len() != pairs.len()
        || keys
            != ["from", "id", "labels", "properties", "to"]
                .into_iter()
                .collect()
    {
        return None;
    }
    pairs.iter().find_map(|(k, val)| match (k, val) {
        (Value::Str(k), Value::Str(id)) if k.as_ref() == "id" => Some(id.as_ref()),
        _ => None,
    })
}

/// Reconstitute an `unfold`ed element column: when every element is a resolvable
/// bare VERTEX (or EDGE) element map (the fold().unfold() round-trip), resolve each
/// `id` back to a live dense id and return a `Col::Nodes` (or `Col::Edges`) so
/// downstream steps operate on the elements again. Otherwise keep the raw `Col::Gen`.
fn reunfold_elements(elems: &[Value], store: &Store) -> Col {
    if elems.is_empty() {
        return Col::Gen(Vec::new());
    }
    let nodes: Option<Vec<u32>> = elems
        .iter()
        .map(|v| vertex_map_ext_id(v).and_then(|ext| store.node_by_ext(ext)))
        .collect();
    if let Some(ids) = nodes {
        return Col::Nodes(ids);
    }
    // Try edges — build a lazy ext→edge map (no reverse map is stored).
    if elems.iter().all(|v| edge_map_ext_id(v).is_some()) {
        let mut by_ext: std::collections::HashMap<Arc<str>, u32> = std::collections::HashMap::new();
        for e in store.all_edges() {
            if let Some(x) = store.edge_ext_id(e) {
                by_ext.entry(x).or_insert(e);
            }
        }
        let eids: Option<Vec<u32>> = elems
            .iter()
            .map(|v| edge_map_ext_id(v).and_then(|ext| by_ext.get(ext).copied()))
            .collect();
        if let Some(eids) = eids {
            return Col::Edges(eids);
        }
    }
    Col::Gen(elems.to_vec())
}

/// The canonical result map for an edge — `{id, from, to, labels(sorted),
/// properties(sorted by key)}`, byte-identical to lenke-core's `val_to_value(Edge)`.
/// `from`/`to` are the endpoint EXTERNAL ids.
fn edge_result_value(store: &Store, eid: u32) -> Value {
    use std::sync::Arc;
    let id = store
        .edge_ext_id(eid)
        .unwrap_or_else(|| Arc::from(format!("e{eid}")));
    let (src, dst) = store.edge_endpoints(eid).unwrap_or((0, 0));
    let ext = |n: u32| {
        store
            .node_ext_id(n)
            .unwrap_or_else(|| Arc::from(n.to_string()))
    };
    // Single edge type here → a one-element (trivially sorted) labels list.
    let labels = Value::List(
        store
            .edge_type_name(eid)
            .into_iter()
            .map(|t| Value::Str(t.into()))
            .collect(),
    );
    let mut props: Vec<(String, Value)> = store
        .edge_prop_keys()
        .into_iter()
        .filter(|k| store.has_edge_prop(eid, k))
        .map(|k| {
            let v = store.edge_prop(eid, &k);
            (k, v)
        })
        .collect();
    props.sort_by(|a, b| a.0.cmp(&b.0));
    let props_map = Value::Map(Arc::new(
        props
            .into_iter()
            .map(|(k, v)| (Value::Str(k.into()), v))
            .collect(),
    ));
    Value::Map(Arc::new(vec![
        (Value::Str("id".into()), Value::Str(id)),
        (Value::Str("from".into()), Value::Str(ext(src))),
        (Value::Str("to".into()), Value::Str(ext(dst))),
        (Value::Str("labels".into()), labels),
        (Value::Str("properties".into()), props_map),
    ]))
}

/// Render one element of an interleaved Gremlin `path()` per its (cycled) `by`
/// modulator. A vertex or an edge (`is_edge`); `Element` → the element map,
/// `Prop` → a property value, `Id`/`Label` → the ext-id / label string.
fn render_gpath_elem(store: &Store, id: u32, is_edge: bool, by: &crate::ir::GPathBy) -> Value {
    use crate::ir::GPathBy;
    match by {
        GPathBy::Element => {
            if is_edge {
                edge_result_value(store, id)
            } else {
                node_result_value(store, id)
            }
        }
        GPathBy::Prop(k) => {
            if is_edge {
                store.edge_prop(id, k)
            } else {
                store.prop(id, k)
            }
        }
        GPathBy::Id => {
            let ext = if is_edge {
                store.edge_ext_id(id)
            } else {
                store.node_ext_id(id)
            };
            ext.map_or(Value::Null, Value::Str)
        }
        GPathBy::Label => {
            if is_edge {
                store
                    .edge_type_name(id)
                    .map_or(Value::Null, |t| Value::Str(t.into()))
            } else {
                store
                    .labels_of(id)
                    .into_iter()
                    .next()
                    .map_or(Value::Null, |l| Value::Str(l.into()))
            }
        }
    }
}

/// Render MANY nodes to their result maps, resolving the property columns ONCE for the
/// whole batch instead of two HashMap-by-key lookups per node per key (what calling
/// [`node_result_value`] per node costs). Byte-identical to per-node rendering: the
/// property map keeps `prop_keys` (sorted) order, filtered to present, and labels stay
/// sorted. The big win for element-materializing shapes — `fold()`, `path`, `valueMap`,
/// `elementMap`, and a bare `g.V()` frontier — where the per-node column re-resolution
/// dominated.
fn render_nodes(store: &Store, ids: &[u32]) -> Vec<Value> {
    use crate::store::Column;
    use std::sync::Arc;
    let keys = store.prop_keys_arc();
    let cols: Vec<(&Arc<str>, &Column)> = keys
        .iter()
        .filter_map(|k| store.column(k).map(|c| (k, c)))
        .collect();
    ids.iter()
        .map(|&id| {
            if id == u32::MAX {
                return Value::Null;
            }
            let i = id as usize;
            let ext = store
                .node_ext_id(id)
                .unwrap_or_else(|| Arc::from(id.to_string()));
            let mut labels = store.labels_of(id);
            labels.sort_unstable();
            let labels_list =
                Value::List(labels.into_iter().map(|l| Value::Str(l.into())).collect());
            let props: Vec<(Value, Value)> = cols
                .iter()
                .filter(|(_, c)| c.present_at(i))
                .map(|(k, c)| (Value::Str(Arc::clone(k)), c.read(i)))
                .collect();
            Value::Map(Arc::new(vec![
                (Value::Str("id".into()), Value::Str(ext)),
                (Value::Str("labels".into()), labels_list),
                (Value::Str("properties".into()), Value::Map(Arc::new(props))),
            ]))
        })
        .collect()
}

/// Resolve the node property columns an element/value map reads, in the SAME order and
/// membership the per-node path produced — sorted keys (every present property, or the
/// `filter` list sorted), each paired with its column. Hoists the per-node
/// `prop_keys()` clone+sort and per-key HashMap probes out of the row loop; the caller
/// then does one `present_at`/`read` per column per node. Byte-identical: `prop_keys_arc`
/// is already sorted, a filter list is sorted here, and a filtered-then-sorted present
/// subset is the same set in the same order.
fn resolve_node_cols<'a>(
    store: &'a Store,
    filter: &[String],
) -> Vec<(std::sync::Arc<str>, &'a crate::store::Column)> {
    use std::sync::Arc;
    if filter.is_empty() {
        store
            .prop_keys_arc()
            .iter()
            .filter_map(|k| store.column(k).map(|c| (Arc::clone(k), c)))
            .collect()
    } else {
        let mut keys = filter.to_vec();
        keys.sort();
        keys.into_iter()
            .filter_map(|k| store.column(&k).map(|c| (Arc::from(k.as_str()), c)))
            .collect()
    }
}

/// The canonical result map for a node — `{id, labels(sorted), properties(sorted by
/// key)}`, byte-identical to lenke-core's `val_to_value(Node)`.
fn node_result_value(store: &Store, id: u32) -> Value {
    use std::sync::Arc;
    let ext = store
        .node_ext_id(id)
        .unwrap_or_else(|| Arc::from(id.to_string()));
    let mut labels = store.labels_of(id);
    labels.sort_unstable();
    let labels_list = Value::List(labels.into_iter().map(|l| Value::Str(l.into())).collect());
    // Present properties on this node, keyed in `prop_keys()` order — which is ALREADY
    // sorted, so the filtered subset stays sorted (core's props_map ordering) with no
    // re-sort and no intermediate Vec.
    let props_map = Value::Map(Arc::new(
        store
            .prop_keys_arc()
            .iter()
            .filter(|k| store.has_prop(id, k))
            .map(|k| {
                let v = store.prop(id, k);
                (Value::Str(Arc::clone(k)), v)
            })
            .collect(),
    ));
    Value::Map(Arc::new(vec![
        (Value::Str("id".into()), Value::Str(ext)),
        (Value::Str("labels".into()), labels_list),
        (Value::Str("properties".into()), props_map),
    ]))
}

/// A self-describing edge record `{id, label, outV, inV, properties}` — the shape
/// core's `subgraph_edge` builds (single `label` string; endpoints as external ids;
/// properties sorted by key).
fn subgraph_edge_value(store: &Store, eid: u32) -> Value {
    use std::sync::Arc;
    let ext = |id: u32| store.node_ext_id(id).map_or(Value::Null, Value::Str);
    let (src, dst) = store.edge_endpoints(eid).unwrap_or((0, 0));
    let mut keys: Vec<String> = store
        .edge_prop_keys()
        .into_iter()
        .filter(|k| store.has_edge_prop(eid, k))
        .collect();
    keys.sort();
    let props: Vec<(Value, Value)> = keys
        .into_iter()
        .map(|k| {
            let v = store.edge_prop(eid, &k);
            (Value::Str(k.into()), v)
        })
        .collect();
    Value::Map(Arc::new(vec![
        (
            Value::Str("id".into()),
            store.edge_ext_id(eid).map_or(Value::Null, Value::Str),
        ),
        (
            Value::Str("label".into()),
            store
                .edge_type_name(eid)
                .map_or(Value::Null, |s| Value::Str(s.into())),
        ),
        (Value::Str("outV".into()), ext(src)),
        (Value::Str("inV".into()), ext(dst)),
        (Value::Str("properties".into()), Value::Map(Arc::new(props))),
    ]))
}

/// The empty result a write statement returns (no columns, no rows).
fn empty_rows() -> Rows {
    Rows {
        names: Vec::new(),
        rows: Flat::default(),
    }
}

/// The output column names a plan produces, seen through row-shape-preserving
/// operators (`Distinct`, `OrderPage`) down to the naming one. `None` means no
/// explicit projection — the row is the raw slot-0 frontier.
fn output_names(plan: &Plan) -> Option<Vec<String>> {
    match plan {
        Plan::Project { items, .. } => Some(items.iter().map(|(n, _)| n.clone()).collect()),
        Plan::Aggregate { keys, aggs, .. } => {
            let mut names: Vec<String> = keys.iter().map(|(n, _)| n.clone()).collect();
            names.extend(aggs.iter().map(|a| a.name.clone()));
            Some(names)
        }
        Plan::Distinct { input }
        | Plan::OrderPage { input, .. }
        | Plan::SortLocal { input, .. } => output_names(input),
        // UNION names come from the LEFT arm (core's rule).
        Plan::Union { left, .. } => output_names(left),
        _ => None,
    }
}

/// Whether any expression in the plan reads the path (`Expr::Path`) — the signal
/// that lineage must be tracked. Computed once, for the whole plan.
fn needs_lineage(plan: &Plan) -> bool {
    fn reads_path(e: &Expr) -> bool {
        match e {
            // Reading any part of the path needs the lineage, just like `Path`.
            Expr::Path | Expr::PathAccess { .. } | Expr::GremlinPath { .. } => true,
            Expr::Compare { left, right, .. }
            | Expr::In {
                needle: left,
                haystack: right,
            } => reads_path(left) || reads_path(right),
            Expr::Not(x) => reads_path(x),
            Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Xor(a, b)
            | Expr::Arith {
                left: a, right: b, ..
            } => reads_path(a) || reads_path(b),
            Expr::Call { args, .. } | Expr::GraphPred { args, .. } | Expr::List { items: args } => {
                args.iter().any(reads_path)
            }
            Expr::Record { fields } | Expr::MapLit { entries: fields } => {
                fields.iter().any(|(_, e)| reads_path(e))
            }
            Expr::Field { base, .. } => reads_path(base),
            Expr::Index { base, index, .. } => reads_path(base) || reads_path(index),
            Expr::Case {
                branches,
                otherwise,
            } => {
                branches.iter().any(|(c, v)| reads_path(c) || reads_path(v))
                    || otherwise.as_deref().is_some_and(reads_path)
            }
            Expr::Cast { expr, .. } | Expr::IsNull { expr, .. } => reads_path(expr),
            // An EXISTS body reads its OWN (sub-)path, never the outer one, and the
            // seed is built without lineage — so it never forces outer tracking.
            Expr::Slot(_)
            | Expr::Prop { .. }
            | Expr::Lit(_)
            | Expr::Param(_)
            | Expr::PropertyExists { .. }
            | Expr::IsLabeled { .. }
            | Expr::Exists { .. }
            | Expr::CountSubquery { .. }
            | Expr::ScalarSubquery { .. }
            | Expr::CollectSubquery { .. }
            | Expr::UncorrelatedExists { .. }
            | Expr::UncorrelatedCount { .. }
            | Expr::UncorrelatedScalar { .. } => false,
        }
    }
    match plan {
        Plan::Scan { .. }
        | Plan::NodeSeed { .. }
        | Plan::EdgeScan
        | Plan::EdgeSeed { .. }
        | Plan::Row
        | Plan::IndexSeek { .. }
        | Plan::RangeSeek { .. }
        | Plan::Insert { .. }
        | Plan::InsertReturn { .. }
        | Plan::Merge { .. }
        | Plan::AddEdge { .. }
        | Plan::CallProcedure { .. }
        | Plan::TxControl { .. }
        | Plan::InsertFrom { .. } => false,
        Plan::Sample { input, .. }
        | Plan::Enumerate { input, .. }
        | Plan::EdgeVertex { input, .. }
        | Plan::Expand { input, .. }
        | Plan::OptionalExpand { input, .. }
        | Plan::VarLength { input, .. }
        | Plan::RepeatGroup { input, .. }
        | Plan::NestedGroup { input, .. }
        | Plan::ShortestPath { input, .. }
        | Plan::Distinct { input }
        | Plan::DistinctBy { input, .. }
        | Plan::Tail { input, .. }
        | Plan::NullPadIfEmpty { input, .. }
        | Plan::GroupToMap { input }
        | Plan::AlgoAnnotate { input, .. }
        | Plan::SortLocal { input, .. } => needs_lineage(input),
        // tree() reads the path lineage itself, so its INPUT must track it.
        Plan::Tree { .. } => true,
        Plan::MapSlot { input, value, .. } => reads_path(value) || needs_lineage(input),
        Plan::Subgraph { input, .. } => needs_lineage(input),
        Plan::ShortestPathEnum { input, .. } => needs_lineage(input),
        Plan::OptionalScan { input, filters, .. } => {
            filters.iter().any(|(_, e)| reads_path(e)) || needs_lineage(input)
        }
        Plan::Unwind { input, list, .. } => reads_path(list) || needs_lineage(input),
        Plan::Branch { input, bodies } => needs_lineage(input) || bodies.iter().any(needs_lineage),
        Plan::IntervalExpand {
            input, qlo, qhi, ..
        } => reads_path(qlo) || reads_path(qhi) || needs_lineage(input),
        Plan::Filter { input, pred } => reads_path(pred) || needs_lineage(input),
        Plan::Project { input, items } => {
            items.iter().any(|(_, e)| reads_path(e)) || needs_lineage(input)
        }
        Plan::Aggregate { input, keys, aggs } => {
            keys.iter().any(|(_, e)| reads_path(e))
                || aggs.iter().any(|a| a.arg.as_ref().is_some_and(reads_path))
                || needs_lineage(input)
        }
        Plan::OrderPage { input, keys, .. } => {
            keys.iter().any(|k| reads_path(&k.expr)) || needs_lineage(input)
        }
        Plan::Join { left, right, .. } | Plan::Union { left, right, .. } => {
            needs_lineage(left) || needs_lineage(right)
        }
        // The subquery yields append columns; whether the OUTER plan needs a path
        // depends on its input (a path read inside the subquery is not surfaced).
        Plan::CallInline { input, yields, .. } => {
            needs_lineage(input) || yields.iter().any(|(_, e)| reads_path(e))
        }
        Plan::Update { input, ops } => {
            needs_lineage(input)
                || ops.iter().any(|op| match op {
                    crate::ir::SetOp::Set { value, .. } => reads_path(value),
                    crate::ir::SetOp::Remove { .. } | crate::ir::SetOp::Delete { .. } => false,
                })
        }
        Plan::UpdateReturn { input, ops, tail } => {
            needs_lineage(input)
                || needs_lineage(tail)
                || ops.iter().any(|op| match op {
                    crate::ir::SetOp::Set { value, .. } => reads_path(value),
                    crate::ir::SetOp::Remove { .. } | crate::ir::SetOp::Delete { .. } => false,
                })
        }
    }
}

/// Pull a batch up through a (non-terminal) plan node. `track` is the plan-global
/// lineage decision: when true, row-producing operators build the path sidecar.
fn pull(plan: &Plan, store: &Store, track: bool) -> Result<Batch, String> {
    Ok(match plan {
        // A write plan is never pulled (a read sub-plan cannot contain one); it
        // is run through `execute`. Yield an empty batch if it somehow reaches
        // here so `run` on a bare write is a harmless no-op rather than a panic.
        Plan::Insert { .. }
        | Plan::InsertFrom { .. }
        | Plan::InsertReturn { .. }
        | Plan::Update { .. }
        | Plan::UpdateReturn { .. }
        | Plan::Merge { .. }
        | Plan::AddEdge { .. }
        | Plan::TxControl { .. } => Batch::of(Vec::new()),
        // `Row` is the leaf of an EXISTS body and is only ever fed a batch by
        // `pull_body`; reaching it through the main pipeline is a bug.
        Plan::Row => {
            // ONE unit row (a single dummy cell so `rows()` == 1) — the input to a
            // bare `RETURN <items>` with no MATCH. A row with no bound variables; the
            // projected items reference no slots. (Inside an EXISTS body, `Plan::Row`
            // is seeded by `pull_body`, not this path.)
            Batch::single(Col::Num(vec![0.0]))
        }
        Plan::Scan { label } => {
            let ids = match label {
                Some(l) => store.nodes_with_label(l).to_vec(),
                None => store.all_nodes(),
            };
            let mut batch = Batch::single(Col::Nodes(ids.clone()));
            if track {
                batch.lineage = Some(Lineage::seed(&ids));
            }
            batch
        }
        Plan::NodeSeed { ext_ids } => {
            // Resolve each external id to a LIVE node; an unknown/deleted id is
            // silently dropped (Gremlin `g.V(<missing>)` yields nothing for it).
            let ids: Vec<u32> = ext_ids
                .iter()
                .filter_map(|e| store.node_by_ext(e).filter(|&id| store.is_alive(id)))
                .collect();
            let mut batch = Batch::single(Col::Nodes(ids.clone()));
            if track {
                batch.lineage = Some(Lineage::seed(&ids));
            }
            batch
        }
        Plan::EdgeScan => {
            // The frontier is EDGES, not nodes. `track` is never set for a bare
            // g.E() read (no path()/lineage step targets an edge frontier yet), so
            // no lineage is seeded here — a path over g.E() is a later item.
            Batch::single(Col::Edges(store.all_edges()))
        }
        Plan::EdgeSeed { ext_ids } => {
            // Resolve each external id to a LIVE edge, preserving request order; an
            // unknown/deleted id is dropped. No reverse ext→edge map exists, so build
            // one lazily from the live-edge set (edge id lookups are rare/small).
            let mut by_ext: std::collections::HashMap<Arc<str>, u32> =
                std::collections::HashMap::new();
            for e in store.all_edges() {
                if let Some(x) = store.edge_ext_id(e) {
                    by_ext.entry(x).or_insert(e);
                }
            }
            let ids: Vec<u32> = ext_ids
                .iter()
                .filter_map(|e| by_ext.get(e.as_str()).copied())
                .collect();
            Batch::single(Col::Edges(ids))
        }
        Plan::Sample { input, n } => {
            // A fixed-seed Mulberry32 partial Fisher-Yates shuffle over the whole row
            // stream, truncated to n — byte-identical to core's sampleStep (same seed,
            // same draw order). The engine's frontier order matches core's here, so the
            // selected subset agrees.
            let b = pull(input, store, track)?;
            let len = b.rows();
            let k = (*n).min(len);
            let mut idx: Vec<usize> = (0..len).collect();
            let mut rng = Mulberry32::new(0x9e37_79b9);
            for i in 0..k {
                let j = i + (rng.next_f64() * (len - i) as f64) as usize;
                idx.swap(i, j);
            }
            idx.truncate(k);
            b.gather(&idx)
        }
        Plan::Enumerate { input, slot } => {
            // Gremlin index(): each row → [element, stream-position]. The element renders
            // as its value (vertices/edges as element maps); position is the row index.
            let b = pull(input, store, track)?;
            let n = b.rows();
            let col = b.slot(*slot);
            let out: Vec<Value> = (0..n)
                .map(|i| Value::List(vec![render_cell(col, i, store), Value::Num(i as f64)]))
                .collect();
            Batch::single(Col::Gen(out))
        }
        Plan::EdgeVertex {
            input,
            edge_slot,
            which,
        } => {
            // Edge frontier → endpoint vertex. Out=src (outV), In=dst (inV), Both=both
            // (fans out to two rows/edge). The endpoint lands in a new appended slot;
            // every other slot is carried through (duplicated for Both).
            let b = pull(input, store, track)?;
            let n = b.rows();
            let mut keep: Vec<usize> = Vec::new();
            let mut nodes: Vec<u32> = Vec::new();
            for i in 0..n {
                let eid = match b.slot(*edge_slot).value_at(i) {
                    Value::Num(x) if x >= 0.0 => x as u32,
                    _ => continue,
                };
                let Some((src, dst)) = store.edge_endpoints(eid) else {
                    continue;
                };
                match which {
                    Dir::Out => {
                        keep.push(i);
                        nodes.push(src);
                    }
                    Dir::In => {
                        keep.push(i);
                        nodes.push(dst);
                    }
                    Dir::Both => {
                        keep.push(i);
                        nodes.push(src);
                        keep.push(i);
                        nodes.push(dst);
                    }
                }
            }
            let mut out = b.gather(&keep);
            out.slots.push(Col::Nodes(nodes));
            out
        }
        Plan::IndexSeek { label, key, value } => {
            let ids = index_seek_ids(store, label, key, value);
            let mut batch = Batch::single(Col::Nodes(ids.clone()));
            if track {
                batch.lineage = Some(Lineage::seed(&ids));
            }
            batch
        }
        Plan::RangeSeek {
            label,
            key,
            op,
            value,
        } => {
            let ids = range_seek_ids(store, label, key, *op, value);
            let mut batch = Batch::single(Col::Nodes(ids.clone()));
            if track {
                batch.lineage = Some(Lineage::seed(&ids));
            }
            batch
        }
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            bind_edge,
            double_loops,
        } => expand(
            &pull(input, store, track)?,
            store,
            *from,
            *dir,
            edge_label,
            *bind_edge,
            *double_loops,
        ),
        Plan::OptionalExpand {
            input,
            from,
            dir,
            edge_label,
            keep_source,
            bind_edge,
        } => optional_expand(
            &pull(input, store, track)?,
            store,
            *from,
            *dir,
            edge_label,
            *keep_source,
            *bind_edge,
        ),
        Plan::IntervalExpand {
            input,
            from,
            dir,
            edge_label,
            lo_key,
            hi_key,
            qlo,
            qhi,
            bind_edge,
        } => {
            let batch = pull(input, store, track)?;
            let qlo_col = eval(qlo, store, &batch)?;
            let qhi_col = eval(qhi, store, &batch)?;
            interval_expand(
                &batch, store, *from, *dir, edge_label, lo_key, hi_key, &qlo_col, &qhi_col,
                *bind_edge,
            )
        }
        Plan::Filter { input, pred } => {
            // Anchor flip: a selective indexed `=` on the traversal TARGET is far
            // cheaper to seed-and-walk-in-reverse than to scan every source and
            // filter. Same slot layout, multiset-preserving.
            if let Some(b) = try_reverse_expand(pred, input, store, track) {
                return Ok(b);
            }
            // Same flip for a VAR-LENGTH hop: seed the selective indexed endpoint and walk
            // the quantified path in reverse, instead of enumerating every forward path
            // (which trips the trail-limit guard on a fanning graph).
            if let Some(b) = try_reverse_varlen(pred, input, store, track) {
                return Ok(b);
            }
            // Target-aware shortest path: a `= t` on the endpoint bounds the BFS to the
            // target's distance (the outer filter here still runs, so this only skips
            // work the filter would discard).
            if let Some(b) = try_shortest_early_stop(pred, input, store, track) {
                return Ok(b);
            }
            let batch = pull(input, store, track)?;
            // Fast path: `<prop> <cmp> <literal>` reads storage in one pass to
            // keep-indices; otherwise evaluate the predicate as a full column.
            let keep: Vec<usize> = match try_filter_keep(pred, store, &batch) {
                Some(keep) => keep,
                None => {
                    // Complex predicate: vectorized three-valued mask (typed numeric leaves,
                    // Kleene AND/OR/NOT), keep the rows that evaluate TRUE.
                    let mask = eval_mask(pred, store, &batch)?;
                    (0..mask.len()).filter(|&i| mask[i] == Some(true)).collect()
                }
            };
            batch.gather(&keep)
        }
        Plan::VarLength {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            mode,
            until,
            body_filter,
            double_loops,
        } => var_length(
            &pull(input, store, track)?,
            store,
            *from,
            *dir,
            edge_label,
            *min,
            *max,
            *mode,
            &[],
            None,
            1,
            until.as_deref(),
            body_filter.as_deref(),
            *double_loops,
        )?,
        Plan::RepeatGroup {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            mode,
            endpoint_slot: _,
            group_binds,
            k,
            per_rep_pred,
        } => var_length(
            &pull(input, store, track)?,
            store,
            *from,
            *dir,
            edge_label,
            *min,
            *max,
            *mode,
            group_binds,
            per_rep_pred.as_deref(),
            *k,
            None,
            None,
            false,
        )?,
        Plan::NestedGroup {
            input,
            from,
            unit,
            min,
            max,
            mode,
            endpoint_slot: _,
            bind_slots,
            per_rep_pred,
        } => nested_group(
            &pull(input, store, track)?,
            store,
            *from,
            unit,
            *min,
            *max,
            *mode,
            bind_slots,
            per_rep_pred.as_deref(),
        ),
        Plan::ShortestPath {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            selector,
            edge_pred,
        } => shortest_path(
            &pull(input, store, track)?,
            store,
            *from,
            *dir,
            edge_label,
            *min,
            *max,
            *selector,
            edge_pred.as_deref(),
            None,
        ),
        Plan::GroupToMap { input } => {
            // Fold the grouped `[key, value]` rows into one Gremlin Map, first-seen
            // key order (the harness compares map content order-independently; core
            // is also first-seen). A single-column value the group produced (count,
            // list) is the map value; a missing second column reads as NULL.
            let b = pull(input, store, track)?;
            let n = b.rows();
            let mut pairs: Vec<(Value, Value)> = Vec::with_capacity(n);
            for i in 0..n {
                let k = b.slots[0].value_at(i);
                let v = b.slots.get(1).map_or(Value::Null, |c| c.value_at(i));
                pairs.push((k, v));
            }
            Batch::single(Col::Gen(vec![Value::Map(std::sync::Arc::new(pairs))]))
        }
        Plan::MapSlot {
            input,
            slot,
            value,
            append,
        } => {
            let mut b = pull(input, store, track)?;
            let col = eval(value, store, &b)?;
            if *append {
                b.slots.push(col);
            } else if *slot < b.slots.len() {
                b.slots[*slot] = col;
            }
            b
        }
        Plan::ShortestPathEnum {
            input,
            node_slot,
            target,
        } => {
            // For each source vertex, emit one row per shortest path (undirected), each
            // a list of the path vertices' external ids (so it compares cleanly — the
            // engine has no Value::Node). Path order is unspecified (multiset). A
            // `with(target, has(…))` keeps only paths whose LAST vertex matches.
            let b = pull(input, store, track)?;
            let n = b.rows();
            let dest_ok = |v: u32| match target {
                None => true,
                Some((key, pred)) => {
                    if !store.has_prop(v, key) {
                        return false;
                    }
                    match pred {
                        None => true,
                        Some((op, want)) => cmp_apply(*op, &store.prop(v, key), want),
                    }
                }
            };
            let mut out: Vec<Value> = Vec::new();
            for i in 0..n {
                let src = match b.slot(*node_slot).value_at(i) {
                    Value::Num(x) if x >= 0.0 => x as u32,
                    _ => continue,
                };
                for path in crate::algo::shortest_paths_from(store, src, crate::ir::Dir::Both) {
                    if !path.last().is_some_and(|&v| dest_ok(v)) {
                        continue;
                    }
                    let ids: Vec<Value> = path
                        .into_iter()
                        .map(|v| store.node_ext_id(v).map_or(Value::Null, Value::Str))
                        .collect();
                    out.push(Value::List(ids));
                }
            }
            Batch::single(Col::Gen(out))
        }
        Plan::Subgraph { input, edge_slot } => {
            // Collect the edge frontier (deduped, first-seen) + their endpoint vertices
            // (deduped), into one {vertices:[…], edges:[…]} Map of element records.
            let b = pull(input, store, track)?;
            let n = b.rows();
            let mut edge_seen: FnvSet<u32> = FnvSet::default();
            let mut vert_seen: FnvSet<u32> = FnvSet::default();
            let mut edges: Vec<Value> = Vec::new();
            let mut verts: Vec<Value> = Vec::new();
            for i in 0..n {
                let eid = match b.slot(*edge_slot).value_at(i) {
                    Value::Num(x) if x >= 0.0 => x as u32,
                    _ => continue,
                };
                if !edge_seen.insert(eid) {
                    continue;
                }
                edges.push(subgraph_edge_value(store, eid));
                if let Some((src, dst)) = store.edge_endpoints(eid) {
                    for v in [src, dst] {
                        if vert_seen.insert(v) {
                            verts.push(node_result_value(store, v));
                        }
                    }
                }
            }
            let map = Value::Map(std::sync::Arc::new(vec![
                (Value::Str("vertices".into()), Value::List(verts)),
                (Value::Str("edges".into()), Value::List(edges)),
            ]));
            Batch::single(Col::Gen(vec![map]))
        }
        Plan::Tree {
            input,
            by,
            leaf_value,
        } => {
            // Fold every traverser's vertex-hop path (node-id lineage) into one nested
            // Map, keyed level-by-level by each element's full element map (bare tree)
            // or its `by` property. Force lineage tracking (Expr::Path reads it).
            let b = pull(input, store, true)?;
            let paths = eval(&Expr::Path, store, &b)?;
            let mut tree = GremlinTree::default();
            for i in 0..b.rows() {
                if let Value::List(ids) = paths.value_at(i) {
                    let mut keys: Vec<Value> = ids
                        .iter()
                        .map(|v| match v {
                            Value::Num(id) => match by {
                                Some(k) => store.prop(*id as u32, k),
                                None => node_result_value(store, *id as u32),
                            },
                            other => other.clone(),
                        })
                        .collect();
                    // A trailing `values('k').tree()` adds a deeper LEAF level keyed by the
                    // last vertex's property (`out(...).values('name').tree()`).
                    if let (Some(lk), Some(Value::Num(last))) = (leaf_value, ids.last()) {
                        keys.push(store.prop(*last as u32, lk));
                    }
                    tree.insert(&keys);
                }
            }
            Batch::single(Col::Gen(vec![tree.to_value()]))
        }
        Plan::AlgoAnnotate {
            input,
            algo,
            edge_label,
            node_slot,
        } => {
            use crate::ir::GremlinAlgo;
            let mut b = pull(input, store, track)?;
            let el = edge_label.as_deref();
            // Compute the per-node result once over the whole store (byte-identical to
            // core — same summation/root rules in `algo`). Component/cluster ids are
            // the ROOT vertex's external-id STRING (core writes the same), pageRank a
            // numeric score.
            let scores: std::collections::HashMap<u32, Value> = match algo {
                GremlinAlgo::PageRank {
                    damping,
                    iterations,
                } => crate::algo::pagerank(store, el, None, *damping, *iterations)
                    .into_iter()
                    .map(|(v, s)| (v, Value::Num(s)))
                    .collect(),
                GremlinAlgo::ConnectedComponent => {
                    crate::algo::weakly_connected_components(store, el)
                        .into_iter()
                        .map(|(v, root)| (v, root_ext_id(store, root)))
                        .collect()
                }
                GremlinAlgo::PeerPressure { iterations } => {
                    crate::algo::peer_pressure(store, el, *iterations)
                        .into_iter()
                        .map(|(v, root)| (v, root_ext_id(store, root)))
                        .collect()
                }
            };
            let n = b.rows();
            let col: Vec<Value> = (0..n)
                .map(|i| match &b.slots[*node_slot] {
                    Col::Nodes(ids) => scores.get(&ids[i]).cloned().unwrap_or(Value::Null),
                    _ => Value::Null,
                })
                .collect();
            b.slots.push(Col::Gen(col));
            b
        }
        Plan::Aggregate { input, keys, aggs } => {
            // REJECTED lever — `WITH <expr> AS a WHERE <pred on a> RETURN count(*)` inlined to
            // one filtered count (fold the alias back, drop the rename projection) so the
            // count ladder could fuse the predicate. Measured a REGRESSION (0.50x -> 0.37x):
            // the WITH's projection NARROWS the batch to one column, which the downstream
            // count rides cheaply; dropping it widens the materialization, and the fused count
            // does not match the combined multi-predicate filter, so nothing is recovered.
            // Frontier fast path: a scalar count over an Expand chain need not
            // build the wide intermediate batch. Falls back to the general
            // aggregate for every shape it does not recognize. (The fused paths
            // never evaluate arbitrary expressions, so they cannot fault.)
            if let Some(b) = try_scan_count(input, keys, aggs, store)
                .or_else(|| try_filtered_count(input, keys, aggs, store))
                .or_else(|| {
                    (keys.is_empty()
                        && aggs.len() == 1
                        && aggs[0].func == AggFn::Count
                        && aggs[0].arg.is_none()
                        && !aggs[0].distinct)
                        .then(|| try_fused_hop_num_count(input, store))
                        .flatten()
                        .map(|c| scalar_num(c as f64))
                })
                .or_else(|| try_fused_hop_mask_agg(input, keys, aggs, store))
                .or_else(|| try_edge_filtered_count(input, keys, aggs, store))
                .or_else(|| try_varlen_count(input, keys, aggs, store))
                .or_else(|| try_edge_cross_count(input, keys, aggs, store))
                .or_else(|| try_frontier_count(input, keys, aggs, store))
                .or_else(|| try_varlen_distinct_count(input, keys, aggs, store))
                .or_else(|| try_varlen_distinctby_count(input, keys, aggs, store))
                .or_else(|| try_varlen_agg(input, keys, aggs, store))
                .or_else(|| try_frontier_prop_agg(input, keys, aggs, store))
                .or_else(|| try_scan_num_agg(input, keys, aggs, store))
                .or_else(|| try_filtered_scan_num_agg(input, keys, aggs, store))
                .or_else(|| try_scan_multi_agg(input, keys, aggs, store))
                .or_else(|| try_scan_distinct_count(input, keys, aggs, store))
                .or_else(|| try_frontier_distinct_count(input, keys, aggs, store))
                .or_else(|| try_3hop_product_count(input, keys, aggs, store))
                .or_else(|| try_fused_count(input, keys, aggs, store))
                .or_else(|| try_node_grouped_count(input, keys, aggs, store))
                .or_else(|| try_scan_dict_count(input, keys, aggs, store))
                .or_else(|| try_frontier_dict_count(input, keys, aggs, store, track))
                .or_else(|| try_scan_group_agg(input, keys, aggs, store))
            {
                b
            } else if let Some(b) = try_frontier_group_fold(input, keys, aggs, store) {
                b
            } else if let Some(b) = try_frontier_aggregate(input, keys, aggs, store)? {
                b
            } else {
                aggregate(&pull(input, store, track)?, store, keys, aggs)?
            }
        }
        Plan::OrderPage {
            input,
            keys,
            skip,
            limit,
        } if *limit == Some(0) => {
            // LIMIT 0 keeps no rows, so the input's projection is never evaluated —
            // a faulting expression (`RETURN 1/0 AS x LIMIT 0`) must yield the empty
            // result, not the fault. Short-circuit without pulling the input. One
            // empty slot keeps the unnamed-output path (`batch.slot(0)`) valid; an
            // empty result carries no column identity anyway.
            let _ = (input, keys, skip);
            Batch::of(vec![Col::Nodes(vec![])])
        }
        Plan::OrderPage {
            input,
            keys,
            skip,
            limit,
        } => {
            // A keyless page (LIMIT/SKIP without ORDER BY) keeps the first
            // `skip+limit` rows in scan order — so cap the input at that many rows
            // instead of materializing the whole scan and slicing. Only safe for a
            // row-preserving Scan/Seek/Project chain (a Filter/Expand would need
            // MORE rows to still yield `limit`), which `pull_capped` recognizes.
            let cap = limit.map(|l| skip.unwrap_or(0).saturating_add(l));
            let capped = match cap {
                Some(c) if keys.is_empty() => match pull_capped(input, store, track, c)? {
                    Some(b) => Some(b),
                    None if track => None,
                    // The row-preserving cap didn't apply (a Filter/Expand/VarLength in
                    // the chain). Stream the chain block-by-block until `c` rows land —
                    // identical rows, computed early. A `DISTINCT … LIMIT` streams with
                    // incremental dedup, stopping at `c` distinct rows.
                    None if reverse_seed_applies(input, store, track) => {
                        // The chain's Filter reverse-seeds a selective indexed endpoint,
                        // so the whole (bounded) result materializes far cheaper than
                        // streaming the forward walk to the cap — which, for an empty or
                        // selective bucket, walks the entire fan-out chasing `limit` rows
                        // that mostly do not exist. Pull it; `order_page` slices. Keyless
                        // page order is unspecified either way, same as the un-capped seed.
                        Some(pull(input, store, track)?)
                    }
                    None => match input.as_ref() {
                        // A `DISTINCT <low-card dict prop> LIMIT n` whose distinct count
                        // cannot reach `n` has a non-binding LIMIT — the capped stream then
                        // scans every block (never hitting the cap) paying per-block dedup,
                        // while the vectorized dedup is far cheaper. Drop to the plain
                        // DISTINCT; `order_page` slices (a no-op). The `+1` leaves room for a
                        // NULL, which DISTINCT counts but the dict does not.
                        Plan::Distinct { input: inner }
                            if distinct_cap_cannot_bind(inner, c, store) =>
                        {
                            Some(pull(input, store, track)?)
                        }
                        Plan::Distinct { input: inner } => {
                            pull_distinct_capped_stream(inner, store, c)?
                        }
                        _ => pull_capped_stream(input, store, c)?,
                    },
                },
                _ => None,
            };
            if let Some(b) = capped {
                order_page(&b, store, keys, *skip, *limit)?
            } else if let Some(b) = try_scan_top_k(input, keys, *skip, *limit, store, track) {
                // Streaming bounded top-K over a bare scan — no full frontier/idx array.
                b
            } else if let Some(b) = try_late_materialize(input, keys, *skip, *limit, store, track)?
            {
                // Sorted top-K over a projection: project only the surviving rows.
                b
            } else {
                order_page(&pull(input, store, track)?, store, keys, *skip, *limit)?
            }
        }
        Plan::Tail { input, n } => {
            // The last `n` rows in input order (Gremlin tail): gather the tail window,
            // computing its start from the materialized row count. `gather` carries
            // the slots AND the lineage sidecar, so a path survives the trim.
            let b = pull(input, store, track)?;
            let rows = b.rows();
            let start = rows.saturating_sub(*n);
            b.gather(&(start..rows).collect::<Vec<usize>>())
        }
        Plan::Branch { input, bodies } => {
            // Gremlin union: run every branch body over the SAME input frontier (each
            // is Row-rooted, correlating on the current slot) and concatenate their
            // sub-rows. Every branch lands its element at the same slot, so the
            // concatenated column keeps its node/edge type — a continuable frontier.
            let inb = pull(input, store, track)?;
            let subs: Vec<Batch> = bodies
                .iter()
                .map(|b| pull_body(b, store, &inb))
                .collect::<Result<_, _>>()?;
            concat_batches(&subs)
        }
        Plan::Project { input, items } => {
            // Fused numeric-filtered projection streams the surviving frontier instead of
            // materializing the whole `[src, nbr]` expand batch (a fixed cost that loses to
            // core's streaming when the filter is mid-selective). Falls through otherwise.
            if let Some(b) = try_fused_hop_project(input, items, store, track) {
                return Ok(b);
            }
            // Project produces a batch whose slots ARE the projected columns, so
            // an operator above it (Distinct, OrderPage) works on the output
            // values, not the pre-projection bindings.
            let batch = pull(input, store, track)?;
            let cols = eval_all(items.iter().map(|(_, e)| e), store, &batch)?;
            Batch::of(cols)
        }
        Plan::Unwind {
            input,
            list,
            var_slot: _,
            ordinal,
        } => {
            // For each input row, evaluate the list and emit one row per element
            // (NULL/empty → none; a non-list scalar → a one-element singleton),
            // appending the element and, optionally, its ordinal counter.
            let batch = pull(input, store, track)?;
            let lists = eval(list, store, &batch)?;
            let mut keep = Vec::new();
            let mut elems: Vec<Value> = Vec::new();
            let mut ords: Vec<Value> = Vec::new();
            for i in 0..batch.rows() {
                let items: Vec<Value> = match lists.value_at(i) {
                    Value::List(v) => v,
                    Value::Null => Vec::new(),
                    scalar => vec![scalar], // a non-list value is a singleton list
                };
                for (j, e) in items.into_iter().enumerate() {
                    keep.push(i);
                    elems.push(e);
                    if let Some((_, one_based)) = ordinal {
                        ords.push(Value::Num((j + usize::from(*one_based)) as f64));
                    }
                }
            }
            let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
            // A folded VERTEX/EDGE round-trips through fold().unfold() as its element
            // map; reconstitute a live element frontier by resolving the `id` field back
            // to a dense id, so `values`/`out`/`order` operate on nodes again. Falls back
            // to the raw map column if any element is not a resolvable element map.
            slots.push(reunfold_elements(&elems, store));
            if ordinal.is_some() {
                slots.push(Col::Gen(ords));
            }
            Batch::of(slots)
        }
        Plan::Union {
            left,
            right,
            all,
            op,
        } => {
            // Run both arms, materialize each row (render_cell → nodes/edges as maps),
            // pad to the LEFT arm's width. UNION concatenates (deduped unless ALL);
            // EXCEPT keeps left rows absent from the right; INTERSECT keeps left rows
            // present in the right (both deduped). Column names come from the left arm.
            let bl = pull(left, store, track)?;
            let br = pull(right, store, track)?;
            let ncols = bl.slots.len();
            // Fast path: UNION ALL of same-width arms concatenates COLUMN-wise.
            if matches!(op, CombineOp::Union) && *all && br.slots.len() == ncols {
                return Ok(concat_batches(&[bl, br]));
            }
            let row_of = |b: &Batch, i: usize| -> Vec<Value> {
                let mut row: Vec<Value> =
                    b.slots.iter().map(|c| render_cell(c, i, store)).collect();
                row.resize(ncols, Value::Null);
                row
            };
            let key_of = |row: &[Value]| -> Vec<u8> {
                let mut buf = Vec::new();
                for v in row {
                    value::group_key_into(v, &mut buf);
                }
                buf
            };
            let rows: Vec<Vec<Value>> = match op {
                CombineOp::Union => {
                    let mut rows: Vec<Vec<Value>> = Vec::with_capacity(bl.rows() + br.rows());
                    for b in [&bl, &br] {
                        for i in 0..b.rows() {
                            rows.push(row_of(b, i));
                        }
                    }
                    if !*all {
                        let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
                        rows.retain(|row| seen.insert(key_of(row)));
                    }
                    rows
                }
                CombineOp::Except | CombineOp::Intersect => {
                    // The right arm's key set; keep a LEFT row iff its key is absent
                    // (EXCEPT) or present (INTERSECT). Always deduped.
                    let mut right_keys: FnvSet<Vec<u8>> = FnvSet::default();
                    for i in 0..br.rows() {
                        right_keys.insert(key_of(&row_of(&br, i)));
                    }
                    let want_present = matches!(op, CombineOp::Intersect);
                    let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
                    let mut rows = Vec::new();
                    for i in 0..bl.rows() {
                        let row = row_of(&bl, i);
                        let k = key_of(&row);
                        if right_keys.contains(&k) == want_present && seen.insert(k) {
                            rows.push(row);
                        }
                    }
                    rows
                }
            };
            let mut cols: Vec<Vec<Value>> = vec![Vec::with_capacity(rows.len()); ncols.max(1)];
            for row in rows {
                for (j, v) in row.into_iter().enumerate() {
                    cols[j].push(v);
                }
            }
            Batch::of(cols.into_iter().map(Col::Gen).collect())
        }
        Plan::Distinct { input } => {
            // Fused `DISTINCT n.k` over a bare Scan: read the storage column and
            // dedup in one pass, emitting ONLY the distinct values — never
            // materializing the 100k-row projected column.
            if let Some(b) = try_distinct_scan_prop(input, store) {
                return Ok(b);
            }
            // The multi-column sibling: dedup several storage columns on a composite
            // key without materializing any of them (the `Arc<str>` dept column above
            // all). Single-column shapes are already handled above; this catches
            // `DISTINCT n.a, n.b, …`.
            if let Some(b) = try_distinct_scan_multi(input, store) {
                return Ok(b);
            }
            // The hop-endpoint sibling: DISTINCT over properties of a chain frontier keys
            // each endpoint node off storage (dict codes), never building the exploded
            // frontier's property columns. Lineage-free (DISTINCT collapses paths).
            if !track {
                // `DISTINCT x, x, …` (every item the identical property) has the same
                // distinct tuples in the same first-seen order as `DISTINCT x` replicated —
                // route to the single-column fast path and clone the output, instead of the
                // composite byte-key (measured ~3x slower on a duplicated Str column).
                if let Some(b) = try_distinct_identical_cols(input, store, track) {
                    return Ok(b);
                }
                if let Some(b) = try_distinct_frontier_prop(input, store) {
                    return Ok(b);
                }
                if let Some(b) = try_distinct_frontier_multi(input, store) {
                    return Ok(b);
                }
                // The expression sibling: `DISTINCT <expr over the endpoint>` (substring,
                // coalesce, arithmetic …) over a var-length hop — dedup endpoints, then
                // project + dedup, never materializing the paths.
                if let Some(b) = try_distinct_varlen_expr(input, store) {
                    return Ok(b);
                }
            }
            distinct_batch(pull(input, store, track)?)
        }
        Plan::DistinctBy { input, key_slots } => {
            // `values(<dict col>).dedup()`: dedup on the dict CODES (≤ dict size) in one
            // pass instead of decoding + hashing every row's string. First-seen by code is
            // the same order as first-seen by string, so byte-identical.
            if key_slots.as_slice() == [0] {
                if let Some(b) = try_distinct_dict_col(input, store) {
                    return Ok(b);
                }
            }
            // Gremlin dedup('a','b'): keep the first row per distinct tuple of the
            // tagged key slots, preserving every other column (group-first-seen keyed
            // on those slots only). Same NaN-never-a-duplicate rule as Distinct.
            // Over a LARGE streamable fan-out, dedup incrementally block-by-block so the
            // exploding frontier is never fully materialized (same first-seen result).
            if let Some(b) = try_distinct_by_streamed(input, key_slots, store, track)? {
                return Ok(b);
            }
            let batch = pull(input, store, track)?;
            let typed = distinct_by_typed(&batch, key_slots);
            let mut seen_ids: FnvSet<u32> = FnvSet::default();
            let mut seen_bytes: FnvSet<Vec<u8>> = FnvSet::default();
            let keep = distinct_by_keep(&batch, key_slots, typed, &mut seen_ids, &mut seen_bytes);
            batch.gather(&keep)
        }
        Plan::SortLocal {
            input,
            descending,
            by_key,
        } => {
            // Gremlin `order(local)`: sort inside each row's slot-0 cell, leaving
            // the batch shape and every other slot untouched. Ordering is the value
            // contract's `cmp_total` (the single home for order); DESC reverses it.
            let batch = pull(input, store, track)?;
            let n = batch.rows();
            let sorted: Vec<Value> = (0..n)
                .map(|i| sort_local_cell(batch.slot(0).value_at(i), *descending, *by_key))
                .collect();
            let mut slots: Vec<Col> = batch.slots.clone();
            if !slots.is_empty() {
                slots[0] = Col::Gen(sorted);
            }
            let mut out = Batch::of(slots);
            out.lineage = batch.lineage;
            out
        }
        Plan::Join { left, right, on } => {
            hash_join(&pull(left, store, track)?, &pull(right, store, track)?, on)
        }
        Plan::NullPadIfEmpty { input, width } => {
            // A leading OPTIONAL MATCH: pass the pattern's rows through, or — when it
            // matched nothing — emit one row of NULL columns. The u32::MAX node
            // sentinel reads back as NULL for any property/element access.
            let batch = pull(input, store, track)?;
            if batch.rows() == 0 {
                Batch::of((0..*width).map(|_| Col::Nodes(vec![u32::MAX])).collect())
            } else {
                batch
            }
        }
        Plan::OptionalScan {
            input,
            label,
            filters,
            node_slot: _,
        } => {
            // A correlated left-outer node lookup: for each input row, the `label` nodes
            // whose prop `k` equals `expr` over that row; else one NULL-node row. The
            // filter exprs are evaluated over the whole input batch (column-at-a-time).
            let batch = pull(input, store, track)?;
            let candidates: Vec<u32> = match label {
                Some(l) => store.nodes_with_label(l).to_vec(),
                None => (0..store.node_count() as u32).collect(),
            };
            let fcols: Vec<Col> = filters
                .iter()
                .map(|(_, e)| eval(e, store, &batch))
                .collect::<Result<_, _>>()?;
            let mut keep: Vec<usize> = Vec::new();
            let mut nodes: Vec<u32> = Vec::new();
            for row in 0..batch.rows() {
                let mut matched = false;
                for &c in &candidates {
                    let ok = filters.iter().zip(&fcols).all(|((key, _), col)| {
                        value::equals(&store.prop(c, key), &col.value_at(row))
                    });
                    if ok {
                        keep.push(row);
                        nodes.push(c);
                        matched = true;
                    }
                }
                if !matched {
                    keep.push(row);
                    nodes.push(u32::MAX); // left-outer: no match → NULL node
                }
            }
            let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
            slots.push(Col::Nodes(nodes));
            Batch::of(slots)
        }
        Plan::CallInline {
            input,
            body,
            yields,
            outer_width,
        } => {
            // Inline correlated (lateral) subquery: run `body` over the outer rows
            // (it is rooted at `Plan::Row`, which yields them), then emit one row
            // per sub-row — the outer slots the sub-row still carries, followed by
            // the yield expressions. Outer rows with no sub-row drop out (inner
            // lateral join). The subquery's internal variables are NOT surfaced.
            let outer = pull(input, store, track)?;
            let ow = *outer_width;
            let sub = pull_body(body, store, &outer)?;
            let mut out_slots: Vec<Col> = (0..ow).map(|i| sub.slot(i).clone()).collect();
            for (_, e) in yields {
                out_slots.push(eval(e, store, &sub)?);
            }
            let mut out = Batch::of(out_slots);
            // Carry any path the sub-rows accumulated (present only under lineage).
            out.lineage = sub.lineage;
            out
        }
        Plan::CallProcedure { name, config } => {
            // A bad config (unknown key / wrong-type value) is a data exception — not
            // silently ignored — matching core.
            crate::algo::validate_config(config)?;
            // Run the named graph algorithm over the whole store into a two-slot
            // batch: node ids, then the per-node result. The parser validates the
            // name, so an unknown one here is defensive.
            let results = crate::algo::run_procedure(store, name, config)
                .ok_or_else(|| format!("unknown procedure `{name}`"))?;
            let ids: Vec<u32> = results.iter().map(|(v, _)| *v).collect();
            // The result column carries per-node Values (a scalar Num for most
            // procedures, a List for neighbor_aggregate's feature vectors).
            let vals: Vec<Value> = results.into_iter().map(|(_, r)| r).collect();
            Batch::of(vec![Col::Nodes(ids), Col::Gen(vals)])
        }
    })
}

/// Hash-join two batches on `(left_slot, right_slot)` key equalities. Output is
/// every left slot gathered by the matched left rows, followed by every right
/// slot gathered by the matched right rows — so right slot `j` lands at output
/// slot `left.len() + j`. Keys are `group_key`-hashed (bound-variable identity),
/// consistent with grouping/distinct.
fn join_key(batch: &Batch, slots: impl Iterator<Item = usize>, row: usize) -> Vec<u8> {
    let mut k = Vec::new();
    for s in slots {
        value::group_key_into(&batch.slot(s).value_at(row), &mut k);
    }
    k
}

fn hash_join(lb: &Batch, rb: &Batch, on: &[(usize, usize)]) -> Batch {
    // Index the right side by its join key.
    let mut index: FnvMap<Vec<u8>, Vec<usize>> = FnvMap::default();
    for j in 0..rb.rows() {
        let k = join_key(rb, on.iter().map(|&(_, r)| r), j);
        index.entry(k).or_default().push(j);
    }
    // Probe with the left side, emitting one combined row per match (a shared key
    // with several matches on each side fans out to their product).
    let mut keep_l = Vec::new();
    let mut keep_r = Vec::new();
    for i in 0..lb.rows() {
        let k = join_key(lb, on.iter().map(|&(l, _)| l), i);
        if let Some(js) = index.get(&k) {
            for &j in js {
                keep_l.push(i);
                keep_r.push(j);
            }
        }
    }
    let mut slots: Vec<Col> = lb.slots.iter().map(|c| c.gather(&keep_l)).collect();
    slots.extend(rb.slots.iter().map(|c| c.gather(&keep_r)));
    Batch::of(slots)
}

/// Pull at most `cap` rows from a ROW-PRESERVING chain (Scan / IndexSeek /
/// RangeSeek / Project over one), so a keyless `LIMIT` need not materialize the
/// whole scan. `Ok(None)` when the input is not cap-safe (a Filter/Expand/Distinct
/// changes the row count, so an input cap could under-produce) — the caller then
/// does the full pull. Faults propagate as `Err`.
fn pull_capped(
    plan: &Plan,
    store: &Store,
    track: bool,
    cap: usize,
) -> Result<Option<Batch>, String> {
    Ok(match plan {
        Plan::Scan { label } => {
            let ids: Vec<u32> = match label {
                Some(l) => store
                    .nodes_with_label(l)
                    .iter()
                    .copied()
                    .take(cap)
                    .collect(),
                None => (0..store.node_count() as u32)
                    .filter(|&i| store.is_alive(i))
                    .take(cap)
                    .collect(),
            };
            let mut b = Batch::single(Col::Nodes(ids.clone()));
            if track {
                b.lineage = Some(Lineage::seed(&ids));
            }
            Some(b)
        }
        Plan::IndexSeek { label, key, value } => {
            let ids: Vec<u32> = index_seek_ids(store, label, key, value)
                .into_iter()
                .take(cap)
                .collect();
            let mut b = Batch::single(Col::Nodes(ids.clone()));
            if track {
                b.lineage = Some(Lineage::seed(&ids));
            }
            Some(b)
        }
        Plan::RangeSeek {
            label,
            key,
            op,
            value,
        } => {
            let ids: Vec<u32> = range_seek_ids(store, label, key, *op, value)
                .into_iter()
                .take(cap)
                .collect();
            let mut b = Batch::single(Col::Nodes(ids.clone()));
            if track {
                b.lineage = Some(Lineage::seed(&ids));
            }
            Some(b)
        }
        Plan::Project { input, items } => match pull_capped(input, store, track, cap)? {
            Some(batch) => {
                let cols = eval_all(items.iter().map(|(_, e)| e), store, &batch)?;
                let mut out = Batch::of(cols);
                out.lineage = batch.lineage;
                Some(out)
            }
            None => None,
        },
        _ => None, // Filter/Expand/Aggregate/Distinct/… change the row count
    })
}

/// If `plan` is a STREAMABLE chain — Project/Filter/Expand/VarLength over a
/// chunkable leaf (Scan/IndexSeek/RangeSeek) — return the chain with the leaf
/// replaced by `Plan::Row` (so it runs from a seeded frontier) plus the leaf's full
/// id list. `None` for any operator the row-by-row `pull_body` cannot stream
/// (Aggregate/Distinct/Join/OrderPage/OptionalExpand/…).
fn streaming_chain(plan: &Plan, store: &Store) -> Option<(Plan, Vec<u32>)> {
    match plan {
        Plan::Scan { label } => {
            let ids: Vec<u32> = match label {
                Some(l) => store.nodes_with_label(l).to_vec(),
                None => (0..store.node_count() as u32)
                    .filter(|&i| store.is_alive(i))
                    .collect(),
            };
            Some((Plan::Row, ids))
        }
        Plan::IndexSeek { label, key, value } => {
            Some((Plan::Row, index_seek_ids(store, label, key, value)))
        }
        Plan::RangeSeek {
            label,
            key,
            op,
            value,
        } => Some((Plan::Row, range_seek_ids(store, label, key, *op, value))),
        Plan::Filter { input, pred } => {
            let (body, ids) = streaming_chain(input, store)?;
            Some((
                Plan::Filter {
                    input: Box::new(body),
                    pred: pred.clone(),
                },
                ids,
            ))
        }
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            bind_edge,
            double_loops,
        } => {
            let (body, ids) = streaming_chain(input, store)?;
            Some((
                Plan::Expand {
                    input: Box::new(body),
                    from: *from,
                    dir: *dir,
                    edge_label: edge_label.clone(),
                    bind_edge: *bind_edge,
                    double_loops: *double_loops,
                },
                ids,
            ))
        }
        Plan::VarLength {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            mode,
            until,
            body_filter,
            double_loops,
        } => {
            let (body, ids) = streaming_chain(input, store)?;
            Some((
                Plan::VarLength {
                    input: Box::new(body),
                    from: *from,
                    dir: *dir,
                    edge_label: edge_label.clone(),
                    min: *min,
                    max: *max,
                    mode: *mode,
                    until: until.clone(),
                    body_filter: body_filter.clone(),
                    double_loops: *double_loops,
                },
                ids,
            ))
        }
        Plan::Project { input, items } => {
            let (body, ids) = streaming_chain(input, store)?;
            Some((
                Plan::Project {
                    input: Box::new(body),
                    items: items.clone(),
                },
                ids,
            ))
        }
        Plan::EdgeVertex {
            input,
            edge_slot,
            which,
        } => {
            let (body, ids) = streaming_chain(input, store)?;
            Some((
                Plan::EdgeVertex {
                    input: Box::new(body),
                    edge_slot: *edge_slot,
                    which: *which,
                },
                ids,
            ))
        }
        _ => None,
    }
}

/// Short-circuit a keyless `LIMIT`/`SKIP` (no `ORDER BY`) over a streamable chain
/// that `pull_capped` can't cap because it filters/expands: run the chain over
/// successive BLOCKS of the source, stopping once `cap` rows have accumulated. The
/// blocks are taken in source-id order and concatenated in order, and per-block
/// operators are the same row-wise ones — so the accumulated rows are IDENTICAL
/// (same order) to materializing the whole input and slicing, just computed early.
/// Only for `!track` (a path-reading LIMIT keeps the full path via the slow path).
fn pull_capped_stream(plan: &Plan, store: &Store, cap: usize) -> Result<Option<Batch>, String> {
    if cap == 0 {
        return Ok(None); // LIMIT 0 handled by the general path (empty, right width)
    }
    let Some((body, ids)) = streaming_chain(plan, store) else {
        return Ok(None);
    };
    if ids.is_empty() {
        return Ok(None); // empty source → let the full path build the right shape
    }
    // ADAPTIVE block size, starting at 1 and doubling. A high-fan-out chain (double
    // var-length) makes even one source overshoot `cap`, so the first block must be
    // tiny — else a fixed block materializes thousands of rows per source. A
    // selective filter / low fan-out grows the block geometrically, so the overhead
    // stays logarithmic. This mirrors a lazy engine producing just past `cap`.
    let mut acc: Vec<Batch> = Vec::new();
    let mut total = 0usize;
    let mut start = 0usize;
    let mut block = 1usize;
    while start < ids.len() && total < cap {
        let end = (start + block).min(ids.len());
        let seed = Batch::single(Col::Nodes(ids[start..end].to_vec()));
        let b = pull_body(&body, store, &seed)?;
        total += b.rows();
        acc.push(b);
        start = end;
        block = block.saturating_mul(2).min(8192);
    }
    Ok(Some(concat_batches(&acc)))
}

/// Streaming `DISTINCT … LIMIT k` over a streamable chain: dedup incrementally
/// (the same whole-row grouping key as `Plan::Distinct`) while streaming source
/// blocks, stopping once `cap` DISTINCT rows are collected. First-occurrence order
/// is preserved (blocks in source-id order), so the result matches a full
/// distinct-then-slice. Lets "give me N distinct X" short-circuit instead of
/// materializing every reachable row before deduping.
fn pull_distinct_capped_stream(
    inner: &Plan,
    store: &Store,
    cap: usize,
) -> Result<Option<Batch>, String> {
    if cap == 0 {
        return Ok(None);
    }
    let Some((body, ids)) = streaming_chain(inner, store) else {
        return Ok(None);
    };
    if ids.is_empty() {
        return Ok(None);
    }
    let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
    let mut buf = Vec::new();
    let mut acc: Vec<Batch> = Vec::new();
    let mut distinct = 0usize;
    let mut start = 0usize;
    let mut block = 1usize;
    while start < ids.len() && distinct < cap {
        let end = (start + block).min(ids.len());
        let b = pull_body(
            &body,
            store,
            &Batch::single(Col::Nodes(ids[start..end].to_vec())),
        )?;
        let mut keep = Vec::new();
        for i in 0..b.rows() {
            buf.clear();
            for c in &b.slots {
                value::group_key_into(&c.value_at(i), &mut buf);
            }
            if !seen.contains(buf.as_slice()) {
                seen.insert(buf.clone());
                keep.push(i);
                distinct += 1;
                if distinct >= cap {
                    break;
                }
            }
        }
        acc.push(b.gather(&keep));
        start = end;
        block = block.saturating_mul(2).min(8192);
    }
    Ok(Some(concat_batches(&acc)))
}

/// The TOP-LEVEL output over a VERY large streamable chain (`<hops>.values(k)` /
/// `.label()` / `.id()`): stream the source in blocks so the exploding per-hop
/// intermediate frontiers are never materialized — expand + project fuse into one pass
/// per block. Only at the top level (never an intermediate projection, which feeds a
/// reducing op that would shrink it anyway) and behind a HIGH size gate, so the block
/// overhead is amortized and medium frontiers stay on the vectorized materialized path.
/// Blocks run in source-id order and concatenate in order → byte-identical. `None`
/// otherwise (small input, lineage, or a non-projection output).
fn pull_top_output_streamed(
    plan: &Plan,
    store: &Store,
    track: bool,
) -> Result<Option<Batch>, String> {
    if track {
        return Ok(None);
    }
    let Plan::Project { input, items } = plan else {
        return Ok(None);
    };
    // A deliberately HIGH threshold: the win is avoiding the per-hop intermediate Cols,
    // which only matters once the fan-out is huge; below it the vectorized materialized
    // project is faster (the block/concat overhead would regress it).
    const STREAM_OUTPUT_ROWS: f64 = 1_000_000.0;
    if crate::cost::estimate(input, store).rows < STREAM_OUTPUT_ROWS {
        return Ok(None);
    }
    let Some((inner_body, ids)) = streaming_chain(input, store) else {
        return Ok(None);
    };
    if ids.is_empty() {
        return Ok(None);
    }
    let body = Plan::Project {
        input: Box::new(inner_body),
        items: items.to_vec(),
    };
    const BLOCK: usize = 8192;
    let mut acc: Vec<Batch> = Vec::new();
    let mut start = 0usize;
    while start < ids.len() {
        let end = (start + BLOCK).min(ids.len());
        acc.push(pull_body(
            &body,
            store,
            &Batch::single(Col::Nodes(ids[start..end].to_vec())),
        )?);
        start = end;
    }
    Ok(Some(concat_batches(&acc)))
}

/// PROTOTYPE streaming result sink (Gremlin single-column array): serialize the result
/// JSON directly, block by block, instead of materializing the whole output `Batch` then
/// the `Rows` then the string. `streaming_chain` runs each hop PER BLOCK, so the full
/// frontier is never built, and only O(block) intermediate values coexist — the win for a
/// projection over a big fan-out and the defuse for a deep-traversal blow-up.
///
/// CONSERVATIVE opt-in: only a terminal single-item `Project` over a streamable chain, and
/// only when the estimated row count clears a high bar (below it the block/serialize
/// overhead loses to the vectorized materialized path). Returns `None` to fall back to
/// `gremlin_results_json(run(plan))`. Byte-identical: same rows in the same
/// (source-id/frontier) order, same per-cell rendering (`col_into_values` +
/// `json::write_value`).
/// Is `body` (the per-block body from `streaming_chain`, with its leaf source replaced
/// by `Plan::Row`) exactly one `Expand` over the block source, carrying no operator that
/// breaks the two invariants the streaming win depends on? That is the only chain whose
/// streamed form reliably beats materializing (measured):
///   - exactly ONE hop — a deeper chain re-runs every hop per block and loses (1.3-7.6x);
///   - the only `Filter` allowed is a bare `PropertyExists` presence check (which
///     `values()` always inserts) — the row estimate the caller gates on models presence
///     accurately, but a VALUE comparison (`has(eq(..))`) it over-counts wildly, which
///     would wave a large regression through the row floor.
fn single_hop_no_filter(body: &Plan) -> bool {
    match body {
        Plan::Project { input, .. } | Plan::EdgeVertex { input, .. } => single_hop_no_filter(input),
        Plan::Filter { input, pred } => {
            matches!(pred, crate::ir::Expr::PropertyExists { .. }) && single_hop_no_filter(input)
        }
        Plan::Expand { input, .. } => matches!(input.as_ref(), Plan::Row),
        _ => false,
    }
}

pub fn try_stream_gremlin_json(
    plan: &Plan,
    store: &Store,
    track: bool,
    min_rows: f64,
) -> Option<String> {
    if track {
        return None; // a path/lineage result is not this shape
    }
    let Plan::Project { input, items } = plan else {
        return None;
    };
    if items.len() != 1 {
        return None; // a Gremlin result is a single column
    }
    // STRUCTURAL gate (measured — see `plan_probe`): stream ONLY the one shape that
    // reliably wins — a SINGLE hop with NO filter. A deeper chain re-runs every hop
    // per block and loses (1.3-7.6x); a filtered chain defeats the row estimate
    // (`has(eq(..))` estimates 144k but matches ~1, so a row gate would wave a 7.6x
    // regression through). Requiring one bare Expand sidesteps both — and there the
    // estimate is trustworthy, so the `min_rows` floor gates the rest.
    let (inner_body, ids) = streaming_chain(input, store)?;
    if !single_hop_no_filter(&inner_body) {
        return None;
    }
    // Deliberately HIGH floor: below it the block/serialize overhead loses to the
    // vectorized materialized path — we do NOT opt in early.
    if crate::cost::estimate(input, store).rows < min_rows {
        return None;
    }
    // Run the chain BELOW the projection per block, then serialize the one output value
    // per row directly — fusing render + serialize so the string heap is touched once,
    // not once to `clone` into a `Col::Str`/`Vec<Value>` and again to serialize. When the
    // sole item is a bare `Prop` over the block's node frontier, `write_nodes_prop_json`
    // reads and writes each property in one scattered pass (no `Arc` bump, no `Value`);
    // any other item falls back to project-then-`write_col_json` (still one pass, no
    // `Vec<Value>`). Byte-identical to `pull_body(Project{..})` then `write_value` each.
    const BLOCK: usize = 8192;
    let single_prop = match items.as_slice() {
        [(_, Expr::Prop { slot, key })] => Some((*slot, key.as_str())),
        _ => None,
    };
    let mut out = String::from("[");
    let mut first = true;
    let mut start = 0usize;
    while start < ids.len() {
        let end = (start + BLOCK).min(ids.len());
        let batch = pull_body(
            &inner_body,
            store,
            &Batch::single(Col::Nodes(ids[start..end].to_vec())),
        )
        .ok()?;
        // Fused fast path: one Prop over a fully-present scalar node column.
        if let Some((slot, key)) = single_prop {
            if let Col::Nodes(nids) = batch.slot(slot) {
                if write_nodes_prop_json(&mut out, store, nids, key, &mut first) {
                    start = end;
                    continue;
                }
            }
        }
        // General path: project the item(s) to one column, serialize it in place.
        let cols = eval_all(items.iter().map(|(_, e)| e), store, &batch).ok()?;
        let col = cols.into_iter().next()?;
        write_col_json(&mut out, &col, store, &mut first);
        start = end;
    }
    out.push(']');
    Some(out)
}

/// Serialize `nids`'s value for property `key` straight into `out` (comma-separated,
/// `first` tracking whether a leading comma is due), when the property is a
/// fully-present scalar column — the fused render+serialize fast path. Returns `false`
/// WITHOUT writing anything when the shape isn't handled (a sentinel id, an absent
/// value, or a non-scalar column), so the caller can fall back for that block. The
/// bytes written are identical to `read_property` → `write_value`: a `Str`/`Dict` cell
/// as a JSON string, `Num` per the number rules, `Bool` as a literal.
fn write_nodes_prop_json(
    out: &mut String,
    store: &Store,
    nids: &[u32],
    key: &str,
    first: &mut bool,
) -> bool {
    if nids.contains(&u32::MAX) {
        return false; // a null sentinel needs the general NULL-carrying path
    }
    let Some(column) = store.column(key) else {
        return false; // missing column → all NULL, let the general path emit nulls
    };
    // All-present check up front: a partial column must not emit a half-written block.
    let present = match column {
        Column::Num { present, .. }
        | Column::Str { present, .. }
        | Column::Dict { present, .. }
        | Column::Bool { present, .. } => present,
        _ => return false,
    };
    if nids.iter().any(|&id| !present[id as usize]) {
        return false;
    }
    let sep = |out: &mut String, first: &mut bool| {
        if !*first {
            out.push(',');
        }
        *first = false;
    };
    match column {
        Column::Str { data, .. } => {
            for &id in nids {
                sep(out, first);
                crate::json::write_string(out, &data[id as usize]);
            }
        }
        Column::Dict { dict, codes, .. } => {
            for &id in nids {
                sep(out, first);
                crate::json::write_string(out, &dict[codes[id as usize] as usize]);
            }
        }
        Column::Num { data, .. } => {
            for &id in nids {
                sep(out, first);
                crate::json::write_value(out, &Value::Num(data[id as usize]));
            }
        }
        Column::Bool { data, .. } => {
            for &id in nids {
                sep(out, first);
                out.push_str(if data[id as usize] { "true" } else { "false" });
            }
        }
        _ => return false,
    }
    true
}

/// Serialize a whole projected `Col` into `out` (comma-separated, `first`-tracked),
/// without the `Col` → `Vec<Value>` step `col_into_values` would take: a typed column
/// (`Str`/`Num`/`Bool`) writes each cell straight, and only a boxed `Gen` column defers
/// to `write_value`. Byte-identical to serializing `col_into_values(col)` cell by cell.
fn write_col_json(out: &mut String, col: &Col, store: &Store, first: &mut bool) {
    let sep = |out: &mut String, first: &mut bool| {
        if !*first {
            out.push(',');
        }
        *first = false;
    };
    match col {
        Col::Str(v) => {
            for s in v {
                sep(out, first);
                crate::json::write_string(out, s);
            }
        }
        Col::Num(v) => {
            for &x in v {
                sep(out, first);
                crate::json::write_value(out, &Value::Num(x));
            }
        }
        Col::Bool(v) => {
            for &b in v {
                sep(out, first);
                out.push_str(if b { "true" } else { "false" });
            }
        }
        // Nodes/Edges render as element maps, Gen carries arbitrary values — both need
        // the full renderer. Reuse `col_into_values` (the identical cells) for these.
        _ => {
            for v in col_into_values(col.clone(), store) {
                sep(out, first);
                crate::json::write_value(out, &v);
            }
        }
    }
}

/// Entry point the FFI's `lnk_e_gremlin_json` uses: stream the result JSON when the shape
/// and cost allow, else materialize + serialize as before. Kept here (not in the FFI) so
/// it has the plan + streaming machinery; falls back transparently.
pub fn run_gremlin_json(plan: &Plan, store: &Store) -> String {
    try_run_gremlin_json(plan, store).expect("read plan evaluation faulted")
}

/// Fallible Gremlin-JSON entry point: an evaluation fault (a bad cast, a cross-type
/// order, …) returns `Err` instead of panicking, so the FFI can surface it as a null
/// result rather than unwinding across the C boundary (which aborts the process).
pub fn try_run_gremlin_json(plan: &Plan, store: &Store) -> Result<String, String> {
    // The fused/streamed sinks below call `pull` directly (off `try_run`'s path), so a
    // var-length traversal here would recurse on the normal stack — route it to the big
    // one, same as `try_run`.
    on_big_stack(plan, || try_run_gremlin_json_inner(plan, store))
}

/// GQL egress (`lnk_e_query_rows`): stream a var-length endpoint projection to the
/// `{columns, rows}` document when it applies — so a large closure completes without
/// materializing the row batch — else materialize + serialize. Big-stack-dispatched like
/// `try_run`. Byte-identical to `gql_rows_json(try_run(...))`.
pub fn try_run_gql_json(plan: &Plan, store: &Store) -> Result<String, String> {
    on_big_stack(plan, || {
        if let Some(res) = try_stream_varlen_json(plan, store, true) {
            return res;
        }
        Ok(crate::json::gql_rows_json(&try_run_inner(plan, store)?))
    })
}

fn try_run_gremlin_json_inner(plan: &Plan, store: &Store) -> Result<String, String> {
    // Fused element/value-map serialization: for a terminal node-map projection, write
    // the JSON straight from the columns and skip building a `Value::Map` tree per row
    // (the dominant cost of these shapes — ~8-10 heap allocs/row otherwise). No cost
    // gate: it is strictly less work than build-then-serialize, so it wins at all sizes.
    if let Some(json) = try_fused_map_json(plan, store) {
        return Ok(json);
    }
    if let Some(json) = try_fused_fold_json(plan, store) {
        return Ok(json);
    }
    if let Some(json) = try_fused_maplit_json(plan, store) {
        return Ok(json);
    }
    // Measured crossover (see `plan_probe`): below ~1M output rows the materialized
    // path ties or wins; the streamed sink pulls 20-29% ahead only at 1M-2.7M+, where
    // it also never builds the full frontier column. Deliberately high — we do NOT opt
    // in eagerly (matches the `pull_top_output_streamed` precedent).
    const STREAM_JSON_ROWS: f64 = 1_000_000.0;
    let track = needs_lineage(plan);
    if let Some(json) = try_stream_gremlin_json(plan, store, track, STREAM_JSON_ROWS) {
        return Ok(json);
    }
    // Stream a var-length endpoint projection straight to JSON — no giant row batch, so a
    // large closure completes (up to a byte cap) where the materialized path would trip
    // the 1M-row trail limit.
    if let Some(res) = try_stream_varlen_json(plan, store, false) {
        return res;
    }
    Ok(crate::json::gremlin_results_json(&try_run(plan, store)?))
}

/// The node-map projection shapes the fused serializer handles, each byte-identical to
/// building the corresponding `Value::Map` then serializing it.
enum NodeMapKind {
    /// A bare node frontier → the NESTED `{id, labels:[…], properties:{…}}` render
    /// (`node_result_value`), where `id` falls back to the dense id as a string.
    Nested,
    /// Gremlin `elementMap()` → the FLAT `{id, label, <props…>}`, where `id`/`label`
    /// are NULL (not a dense-id fallback) when absent.
    Flat,
    /// Gremlin `valueMap()` (`wrap=false`) / `propertyMap()` (`wrap=true`, each value in
    /// a one-element list) → just the present properties.
    Value { wrap: bool },
}

/// `g.V().project(k…).by(e…)` and GQL map projections — a terminal `Project{[MapLit]}`
/// — serialized straight to `[{k:v,…},…]`, skipping the per-row `Value::Map` (its Vec,
/// Arc, and freshly-allocated key Arcs). Values are computed vectorized (`eval_all`) once;
/// a scalar cell writes directly, an element cell falls back to `render_cell` (its element
/// map). Byte-identical to building the map then serializing: same key order, same values.
fn try_fused_maplit_json(plan: &Plan, store: &Store) -> Option<String> {
    let Plan::Project { input, items } = plan else {
        return None;
    };
    let [(_, Expr::MapLit { entries })] = items.as_slice() else {
        return None;
    };
    if needs_lineage(plan) {
        return None;
    }
    let batch = pull(input, store, false).ok()?;
    let cols = eval_all(entries.iter().map(|(_, e)| e), store, &batch).ok()?;
    let mut out = String::from("[");
    for i in 0..batch.rows() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        for (j, ((k, _), col)) in entries.iter().zip(&cols).enumerate() {
            if j > 0 {
                out.push(',');
            }
            crate::json::write_string(&mut out, k);
            out.push(':');
            match col {
                Col::Str(v) => crate::json::write_string(&mut out, &v[i]),
                Col::Num(v) => crate::json::write_value(&mut out, &Value::Num(v[i])),
                Col::Bool(v) => out.push_str(if v[i] { "true" } else { "false" }),
                other => crate::json::write_value(&mut out, &render_cell(other, i, store)),
            }
        }
        out.push('}');
    }
    out.push(']');
    Some(out)
}

/// `g.V().fold()` / `g.E().fold()` — a whole node/edge frontier collected into ONE list —
/// serialized straight to `[[<element maps>]]`, skipping the `Value::List` of `Value::Map`
/// trees the general `Collect` aggregate builds (the same per-node allocation the terminal
/// map writer eliminates, here for the list-wrapped fold). Only a keyless, non-distinct
/// `Collect(Slot)` over a node/edge frontier; anything else falls back.
fn try_fused_fold_json(plan: &Plan, store: &Store) -> Option<String> {
    let Plan::Aggregate { input, keys, aggs } = plan else {
        return None;
    };
    if !keys.is_empty() || aggs.len() != 1 || needs_lineage(plan) {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Collect || agg.distinct {
        return None;
    }
    let Some(Expr::Slot(s)) = &agg.arg else {
        return None;
    };
    let batch = pull(input, store, false).ok()?;
    let cols = resolve_node_cols(store, &[]);
    let mut out = String::from("[[");
    match batch.slot(*s) {
        Col::Nodes(ids) => {
            for (n, &id) in ids.iter().enumerate() {
                if n > 0 {
                    out.push(',');
                }
                if id == u32::MAX {
                    out.push_str("null");
                } else {
                    write_node_nested_map(&mut out, store, id, &cols);
                }
            }
        }
        Col::Edges(eids) => {
            let ecols = resolve_edge_cols(store, &[]);
            for (n, &eid) in eids.iter().enumerate() {
                if n > 0 {
                    out.push(',');
                }
                if eid == u32::MAX {
                    out.push_str("null");
                } else {
                    write_edge_nested_map(&mut out, store, eid, &ecols);
                }
            }
        }
        _ => return None, // a folded scalar list is not an element-map fold
    }
    out.push_str("]]");
    Some(out)
}

/// Serialize a terminal single-column node-map projection directly to JSON, skipping the
/// per-row `Value::Map` tree. Returns `None` (→ the caller's slower path) for anything
/// not a node frontier rendered as an element/value map: an edge frontier, a scalar
/// projection, a lineage-tracked plan, or a non-`Slot` map argument.
fn try_fused_map_json(plan: &Plan, store: &Store) -> Option<String> {
    if needs_lineage(plan) {
        return None;
    }
    // `input` is the plan whose batch to pull; `slot` the frontier column within it.
    let (input, kind, slot, filter): (&Plan, NodeMapKind, usize, Vec<String>) = match plan {
        Plan::Project { input, items } if items.len() == 1 => match &items[0].1 {
            // A bare frontier projection renders as the nested element map (render_cell).
            Expr::Slot(s) => (input.as_ref(), NodeMapKind::Nested, *s, Vec::new()),
            Expr::Call { name, args }
                if matches!(name.as_str(), "element_map" | "value_map" | "property_map") =>
            {
                let Some(Expr::Slot(s)) = args.first() else {
                    return None; // a non-slot element arg (rare) keeps the general path
                };
                let filter = args[1..]
                    .iter()
                    .filter_map(|e| match e {
                        Expr::Lit(Value::Str(s)) => Some(s.to_string()),
                        _ => None,
                    })
                    .collect();
                let kind = match name.as_str() {
                    "element_map" => NodeMapKind::Flat,
                    "value_map" => NodeMapKind::Value { wrap: false },
                    _ => NodeMapKind::Value { wrap: true },
                };
                (input.as_ref(), kind, *s, filter)
            }
            _ => return None,
        },
        // A projection-LESS read frontier (`g.V()`, `g.E()`, a bare filtered/dedup'd
        // frontier): `try_run` renders slot 0 as its nested element map when
        // `output_names` is None. Mirror that exact single-column path, fused.
        other if output_names(other).is_none() => (other, NodeMapKind::Nested, 0, Vec::new()),
        _ => return None,
    };
    let batch = pull(input, store, false).ok()?;
    let mut out = String::from("[");
    match batch.slot(slot) {
        Col::Nodes(ids) => {
            let cols = resolve_node_cols(store, &filter);
            for (n, &id) in ids.iter().enumerate() {
                if n > 0 {
                    out.push(',');
                }
                if id == u32::MAX {
                    out.push_str("null"); // the OPTIONAL-match null sentinel
                    continue;
                }
                match &kind {
                    NodeMapKind::Nested => write_node_nested_map(&mut out, store, id, &cols),
                    NodeMapKind::Flat => write_node_flat_map(&mut out, store, id, &cols),
                    NodeMapKind::Value { wrap } => write_node_value_map(&mut out, id, &cols, *wrap),
                }
            }
        }
        Col::Edges(eids) => {
            let cols = resolve_edge_cols(store, &filter);
            for (n, &eid) in eids.iter().enumerate() {
                if n > 0 {
                    out.push(',');
                }
                if eid == u32::MAX {
                    out.push_str("null");
                    continue;
                }
                match &kind {
                    NodeMapKind::Nested => write_edge_nested_map(&mut out, store, eid, &cols),
                    NodeMapKind::Flat => write_edge_flat_map(&mut out, store, eid, &cols),
                    NodeMapKind::Value { wrap } => {
                        write_edge_value_map(&mut out, eid, &cols, *wrap)
                    }
                }
            }
        }
        // A scalar column (e.g. a projected value) is not an element map — fall back.
        _ => return None,
    }
    out.push(']');
    Some(out)
}

/// A resolved read handle for one edge property — the dense numeric overlay when fresh,
/// else the boxed per-eid map. `cell(eid)` returns the present value or `None`, matching
/// `store.edge_prop`/`has_edge_prop` byte-for-byte (the overlay is kept in step with the
/// boxed source, and only homogeneously-numeric keys get one).
enum EdgeCol<'a> {
    Num(&'a [f64], &'a [bool]),
    Boxed(&'a crate::store::EdgeMap),
}

impl EdgeCol<'_> {
    fn cell(&self, eid: u32) -> Option<Value> {
        match self {
            EdgeCol::Num(data, present) => {
                let i = eid as usize;
                (i < present.len() && present[i]).then(|| Value::Num(data[i]))
            }
            EdgeCol::Boxed(map) => map.get(&eid).cloned(),
        }
    }
}

/// Resolve the edge property read handles once, in the SAME sorted-key order and
/// membership the per-edge path produced — hoisting `edge_prop_keys()` clone+sort and
/// the per-key `edge_prop_map`/`edge_num_column` lookups out of the row loop.
fn resolve_edge_cols<'a>(
    store: &'a Store,
    filter: &[String],
) -> Vec<(std::sync::Arc<str>, EdgeCol<'a>)> {
    use std::sync::Arc;
    let keys: Vec<String> = if filter.is_empty() {
        store.edge_prop_keys() // already sorted
    } else {
        let mut k = filter.to_vec();
        k.sort();
        k
    };
    keys.into_iter()
        .filter_map(|k| {
            let col = match store.edge_num_column(&k) {
                Some((d, p)) => EdgeCol::Num(d, p),
                None => EdgeCol::Boxed(store.edge_prop_map(&k)?),
            };
            Some((Arc::from(k.as_str()), col))
        })
        .collect()
}

/// A node's external id as a JSON string, falling back to its dense id (the `id`/`from`/
/// `to` rule for the NESTED renders).
fn write_node_ext_or_dense(out: &mut String, store: &Store, n: u32) {
    match store.node_ext_id(n) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => crate::json::write_string(out, &n.to_string()),
    }
}

/// A `{id, label}` endpoint stub for the flat edge `elementMap` — `id`/`label` NULL when
/// absent (matching the `node_id`/`node_label` closures).
fn write_node_stub(out: &mut String, store: &Store, v: u32) {
    out.push_str("{\"id\":");
    match store.node_ext_id(v) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => out.push_str("null"),
    }
    out.push_str(",\"label\":");
    match store.labels_of(v).first() {
        Some(l) => crate::json::write_string(out, l),
        None => out.push_str("null"),
    }
    out.push('}');
}

/// `{id, from, to, labels:[type?], properties:{sorted present}}` — the nested edge render,
/// byte-identical to `edge_result_value(store, eid)` serialized.
fn write_edge_nested_map(
    out: &mut String,
    store: &Store,
    eid: u32,
    cols: &[(std::sync::Arc<str>, EdgeCol<'_>)],
) {
    out.push_str("{\"id\":");
    match store.edge_ext_id(eid) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => crate::json::write_string(out, &format!("e{eid}")),
    }
    let (src, dst) = store.edge_endpoints(eid).unwrap_or((0, 0));
    out.push_str(",\"from\":");
    write_node_ext_or_dense(out, store, src);
    out.push_str(",\"to\":");
    write_node_ext_or_dense(out, store, dst);
    out.push_str(",\"labels\":[");
    if let Some(t) = store.edge_type_name(eid) {
        crate::json::write_string(out, &t);
    }
    out.push_str("],\"properties\":{");
    let mut first = true;
    for (k, col) in cols {
        if let Some(v) = col.cell(eid) {
            if !first {
                out.push(',');
            }
            first = false;
            crate::json::write_string(out, k);
            out.push(':');
            crate::json::write_value(out, &v);
        }
    }
    out.push_str("}}");
}

/// `{id, label, IN:{…}, OUT:{…}, <sorted props flat>}` — the flat Gremlin `elementMap()`
/// on an edge (IN is the destination, OUT the source; `id`/`label` NULL when absent).
fn write_edge_flat_map(
    out: &mut String,
    store: &Store,
    eid: u32,
    cols: &[(std::sync::Arc<str>, EdgeCol<'_>)],
) {
    out.push_str("{\"id\":");
    match store.edge_ext_id(eid) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => out.push_str("null"),
    }
    out.push_str(",\"label\":");
    match store.edge_type_name(eid) {
        Some(t) => crate::json::write_string(out, &t),
        None => out.push_str("null"),
    }
    if let Some((src, dst)) = store.edge_endpoints(eid) {
        out.push_str(",\"IN\":");
        write_node_stub(out, store, dst);
        out.push_str(",\"OUT\":");
        write_node_stub(out, store, src);
    }
    for (k, col) in cols {
        if let Some(v) = col.cell(eid) {
            out.push(',');
            crate::json::write_string(out, k);
            out.push(':');
            crate::json::write_value(out, &v);
        }
    }
    out.push('}');
}

/// `{sorted present edge props}` — Gremlin `valueMap()`/`propertyMap()` on an edge.
fn write_edge_value_map(
    out: &mut String,
    eid: u32,
    cols: &[(std::sync::Arc<str>, EdgeCol<'_>)],
    wrap: bool,
) {
    out.push('{');
    let mut first = true;
    for (k, col) in cols {
        if let Some(v) = col.cell(eid) {
            if !first {
                out.push(',');
            }
            first = false;
            crate::json::write_string(out, k);
            out.push(':');
            if wrap {
                out.push('[');
                crate::json::write_value(out, &v);
                out.push(']');
            } else {
                crate::json::write_value(out, &v);
            }
        }
    }
    out.push('}');
}

/// Write one present property column cell straight to `out` (the caller guarantees
/// `present_at(i)`), avoiding the `Arc`/`Value` a `Column::read` would build for the
/// scalar cases. Byte-identical to `write_value(&col.read(i))`.
fn write_col_cell_json(out: &mut String, col: &crate::store::Column, i: usize) {
    use crate::store::Column;
    match col {
        Column::Str { data, .. } => crate::json::write_string(out, &data[i]),
        Column::Dict { dict, codes, .. } => {
            crate::json::write_string(out, &dict[codes[i] as usize]);
        }
        Column::Num { data, .. } => crate::json::write_value(out, &Value::Num(data[i])),
        Column::Bool { data, .. } => out.push_str(if data[i] { "true" } else { "false" }),
        // Temporal / Gen: defer to the value renderer (the leaf types the fast path skips).
        other => crate::json::write_value(out, &other.read(i)),
    }
}

/// `{id, labels:[sorted], properties:{sorted present}}` — the nested node render, written
/// directly. Byte-identical to `node_result_value(store, id)` serialized.
fn write_node_nested_map(
    out: &mut String,
    store: &Store,
    id: u32,
    cols: &[(std::sync::Arc<str>, &crate::store::Column)],
) {
    let i = id as usize;
    out.push_str("{\"id\":");
    match store.node_ext_id(id) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => crate::json::write_string(out, &id.to_string()),
    }
    out.push_str(",\"labels\":[");
    for (j, l) in store.labels_of(id).iter().enumerate() {
        if j > 0 {
            out.push(',');
        }
        crate::json::write_string(out, l);
    }
    out.push_str("],\"properties\":{");
    let mut first = true;
    for (k, col) in cols {
        if col.present_at(i) {
            if !first {
                out.push(',');
            }
            first = false;
            crate::json::write_string(out, k);
            out.push(':');
            write_col_cell_json(out, col, i);
        }
    }
    out.push_str("}}");
}

/// `{id, label, <sorted present props flat>}` — the flat Gremlin `elementMap()` shape.
/// `id`/`label` are NULL when absent (no dense-id fallback, unlike the nested render).
fn write_node_flat_map(
    out: &mut String,
    store: &Store,
    id: u32,
    cols: &[(std::sync::Arc<str>, &crate::store::Column)],
) {
    let i = id as usize;
    out.push_str("{\"id\":");
    match store.node_ext_id(id) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => out.push_str("null"),
    }
    out.push_str(",\"label\":");
    match store.labels_of(id).first() {
        Some(l) => crate::json::write_string(out, l),
        None => out.push_str("null"),
    }
    // id and label are always emitted, so every property is comma-prefixed.
    for (k, col) in cols {
        if col.present_at(i) {
            out.push(',');
            crate::json::write_string(out, k);
            out.push(':');
            write_col_cell_json(out, col, i);
        }
    }
    out.push('}');
}

/// `{sorted present props}` — Gremlin `valueMap()` (`wrap=false`) or `propertyMap()`
/// (`wrap=true`, each value wrapped in a one-element list).
fn write_node_value_map(
    out: &mut String,
    id: u32,
    cols: &[(std::sync::Arc<str>, &crate::store::Column)],
    wrap: bool,
) {
    let i = id as usize;
    out.push('{');
    let mut first = true;
    for (k, col) in cols {
        if col.present_at(i) {
            if !first {
                out.push(',');
            }
            first = false;
            crate::json::write_string(out, k);
            out.push(':');
            if wrap {
                out.push('[');
                write_col_cell_json(out, col, i);
                out.push(']');
            } else {
                write_col_cell_json(out, col, i);
            }
        }
    }
    out.push('}');
}

/// Keep the rows of `batch` whose key (the `key_slots` tuple) is seen for the FIRST
/// time, threading the seen-set across calls (streamed dedup). A SINGLE node/edge key
/// slot (`typed`) deduplicates by raw `u32` id — no per-row byte-key serialization,
/// which dominated `dedup()` over a big fan-out; `value_at(Col::Nodes)` is `Node(id)`
/// so a node's group key IS its id, keeping it byte-identical.
fn distinct_by_keep(
    batch: &Batch,
    key_slots: &[usize],
    typed: bool,
    seen_ids: &mut FnvSet<u32>,
    seen_bytes: &mut FnvSet<Vec<u8>>,
) -> Vec<usize> {
    let mut keep = Vec::new();
    if typed {
        let ids: &[u32] = match batch.slot(key_slots[0]) {
            Col::Nodes(v) | Col::Edges(v) => v,
            _ => unreachable!("typed only when the single key slot is Nodes/Edges"),
        };
        for (i, &id) in ids.iter().enumerate() {
            if seen_ids.insert(id) {
                keep.push(i);
            }
        }
    } else {
        let mut buf = Vec::new();
        for i in 0..batch.rows() {
            buf.clear();
            for &s in key_slots {
                value::group_key_into(&batch.slot(s).value_at(i), &mut buf);
            }
            if seen_bytes.insert(buf.clone()) {
                keep.push(i);
            }
        }
    }
    keep
}

/// Whether `batch`'s single key slot is a node/edge column (so `distinct_by_keep` can
/// key by raw `u32`).
fn distinct_by_typed(batch: &Batch, key_slots: &[usize]) -> bool {
    key_slots.len() == 1 && matches!(batch.slot(key_slots[0]), Col::Nodes(_) | Col::Edges(_))
}

/// UNCAPPED `dedup` (`Plan::DistinctBy`) over a streamable chain with BOUNDED memory:
/// stream the source in blocks through the chain, deduping incrementally on `key_slots`
/// into a global key set and keeping only the first occurrence of each key. A high
/// fan-out (`both().both()…`) that dedups down to ≤ node_count distinct keys never
/// materializes the exploding frontier — the peak is one block's expansion plus the
/// distinct rows. Blocks run in source-id order, so first-occurrence order (hence the
/// result) is byte-identical to materialize-then-dedup. Gated to a large estimated
/// input and `!track` (lineage would need the full path); `None` otherwise.
fn try_distinct_by_streamed(
    inner: &Plan,
    key_slots: &[usize],
    store: &Store,
    track: bool,
) -> Result<Option<Batch>, String> {
    if track {
        return Ok(None); // a path-reading dedup keeps the full lineage — slow path
    }
    if !crate::cost::prefer_bounded_memory(inner, store, &crate::cost::Budget::default_budget()) {
        return Ok(None); // small intermediate → materializing is cheaper than blocking
    }
    let Some((body, ids)) = streaming_chain(inner, store) else {
        return Ok(None);
    };
    if ids.is_empty() {
        return Ok(None);
    }
    // A fixed block bounds the peak intermediate (block × fan-out) without the
    // early-stop adaptive sizing the capped streamers need (here we scan everything).
    const BLOCK: usize = 2048;
    let mut seen_ids: FnvSet<u32> = FnvSet::default();
    let mut seen_bytes: FnvSet<Vec<u8>> = FnvSet::default();
    let mut typed: Option<bool> = None;
    let mut acc: Vec<Batch> = Vec::new();
    let mut start = 0usize;
    while start < ids.len() {
        let end = (start + BLOCK).min(ids.len());
        let b = pull_body(
            &body,
            store,
            &Batch::single(Col::Nodes(ids[start..end].to_vec())),
        )?;
        let t = *typed.get_or_insert_with(|| distinct_by_typed(&b, key_slots));
        let keep = distinct_by_keep(&b, key_slots, t, &mut seen_ids, &mut seen_bytes);
        acc.push(b.gather(&keep));
        start = end;
    }
    Ok(Some(concat_batches(&acc)))
}

/// Compare two rows by the sort keys only (`Equal` on a full tie). NULL placement
/// is the front-end's language contract (GQL last, Gremlin first), decided here
/// BEFORE the total order (not by reversing `cmp_total` under DESC).
#[inline]
fn row_cmp(
    key_cols: &[Col],
    keys: &[crate::ir::SortKey],
    a: usize,
    b: usize,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for (kc, k) in key_cols.iter().zip(keys) {
        let (va, vb) = (kc.value_at(a), kc.value_at(b));
        let ord = match (va.is_null(), vb.is_null()) {
            (true, true) => Ordering::Equal,
            (true, false) => {
                if k.nulls_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            (false, true) => {
                if k.nulls_first {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            (false, false) => {
                let o = value::cmp_total(&va, &vb);
                if k.descending {
                    o.reverse()
                } else {
                    o
                }
            }
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Order `idx` (initially `0..idx.len()`) by `keys` so that `idx[..end]` are the
/// `end` rows the sort window needs, correctly ordered. When `end < n` only the
/// top-`end` are selected (partition + partial sort — O(n) + O(end log end)); the
/// full case keeps a stable sort (ties keep arrival order). A single numeric key
/// compares raw f64 via `cmp_num_total` (no per-comparison `value_at` boxing).
fn sort_idx(idx: &mut [usize], key_cols: &[Col], keys: &[crate::ir::SortKey], end: usize) {
    let n = idx.len();
    if keys.is_empty() {
        return;
    }
    if let [Col::Num(vals)] = key_cols {
        // Col::Num carries no nulls, so null placement is moot; an arrival-index
        // tiebreak makes the order total (deterministic, == the stable sort on ties).
        // Sort (key, index) PAIRS so each comparison reads the f64 INLINE — the index-only
        // sort loads `vals[a]`/`vals[b]` (random) on every compare and cache-misses; carrying
        // the key with the index does one gather, then compares locally. Same value-then-
        // index order → byte-identical.
        let desc = keys[0].descending;
        let mut pairs: Vec<(f64, usize)> = idx.iter().map(|&i| (vals[i], i)).collect();
        let cmp = |a: &(f64, usize), b: &(f64, usize)| {
            let o = if desc {
                value::cmp_num_total(b.0, a.0)
            } else {
                value::cmp_num_total(a.0, b.0)
            };
            o.then(a.1.cmp(&b.1))
        };
        if end < n {
            pairs.select_nth_unstable_by(end - 1, cmp);
            pairs[..end].sort_unstable_by(cmp);
        } else {
            pairs.sort_unstable_by(cmp);
        }
        for (slot, p) in idx.iter_mut().zip(&pairs) {
            *slot = p.1;
        }
    } else if let [Col::Str(vals)] = key_cols {
        // A single string key: compare the `Arc<str>` cells directly (lexicographic,
        // == `cmp_total` for strings) instead of boxing each cell through `value_at`.
        // Col::Str carries no nulls; the arrival-index tiebreak keeps it total/stable.
        let desc = keys[0].descending;
        let cmp = |&a: &usize, &b: &usize| {
            let o = if desc {
                vals[b].as_ref().cmp(vals[a].as_ref())
            } else {
                vals[a].as_ref().cmp(vals[b].as_ref())
            };
            o.then(a.cmp(&b))
        };
        if end < n {
            idx.select_nth_unstable_by(end - 1, cmp);
            idx[..end].sort_unstable_by(cmp);
        } else {
            idx.sort_unstable_by(cmp);
        }
    } else if end < n {
        let total = |&a: &usize, &b: &usize| row_cmp(key_cols, keys, a, b).then(a.cmp(&b));
        idx.select_nth_unstable_by(end - 1, total);
        idx[..end].sort_unstable_by(total);
    } else {
        idx.sort_by(|&a, &b| row_cmp(key_cols, keys, a, b));
    }
}

/// STREAMING TOP-K for `ORDER BY <numeric prop> [DESC] LIMIT k` over a bare `Scan`:
/// instead of materializing the whole frontier + key column + an index array and then
/// partial-sorting (what `order_page` does on a pulled batch), scan the nodes once,
/// reading the key sequentially, and keep only the best `skip+limit` in a bounded buffer
/// (periodically trimmed with `select_nth`). O(N) time, O(k) space — matching core's
/// streaming heap. Returns the top-K rows as the OrderPage output (a single `Col::Nodes`
/// in sort order), so a `Project` above it builds its columns for K rows, not N.
///
/// `None` (fall back to `order_page`) for anything outside the narrow shape: lineage
/// tracking, a non-`Scan` input, a multi-key / non-`Prop(slot 0)` sort, a non-numeric or
/// null-bearing key column (the ordering with nulls goes through the general path), or a
/// window that is not a small prefix (streaming buys nothing near a full sort).
fn try_scan_top_k(
    input: &Plan,
    keys: &[crate::ir::SortKey],
    skip: Option<usize>,
    limit: Option<usize>,
    store: &Store,
    track: bool,
) -> Option<Batch> {
    if track {
        return None; // a path/lineage sort is not this shape
    }
    let limit = limit?;
    let [k] = keys else { return None };
    let Expr::Prop { slot: 0, key } = &k.expr else {
        return None;
    };
    let Plan::Scan { label } = input else {
        return None;
    };
    let Some(Column::Num { data, present, .. }) = store.column(key) else {
        return None; // numeric, unboxed key only (matches order_page's Col::Num path)
    };
    let kcap = skip.unwrap_or(0).checked_add(limit)?;
    if kcap == 0 {
        return Some(Batch::of(vec![Col::Nodes(Vec::new())]));
    }
    if kcap.saturating_mul(2) >= store.node_count() {
        return None; // window is not a small prefix — a full sort is as cheap
    }
    let desc = k.descending;
    // Sort order as `order_page`'s Col::Num path: key asc/desc, then arrival (row order)
    // ascending as a total tiebreak. `cmp` "less" = ranks earlier = keep.
    let cmp = |a: &(f64, u32, u32), b: &(f64, u32, u32)| {
        let o = if desc {
            value::cmp_num_total(b.0, a.0)
        } else {
            value::cmp_num_total(a.0, b.0)
        };
        o.then(a.1.cmp(&b.1))
    };
    let trim = kcap.saturating_mul(4).max(1024);
    let mut buf: Vec<(f64, u32, u32)> = Vec::with_capacity(trim);
    let mut arrival = 0u32;
    let mut has_null = false;
    scan_visit(store, label, |i| {
        if !present[i] {
            has_null = true; // key ordering with nulls → general path
            return;
        }
        buf.push((data[i], arrival, i as u32));
        arrival += 1;
        if buf.len() >= trim {
            buf.select_nth_unstable_by(kcap - 1, cmp);
            buf.truncate(kcap);
        }
    });
    if has_null {
        return None;
    }
    let end = kcap.min(buf.len());
    if end < buf.len() {
        buf.select_nth_unstable_by(end - 1, cmp);
        buf.truncate(end);
    }
    buf.sort_unstable_by(cmp);
    let start = skip.unwrap_or(0).min(buf.len());
    Some(Batch::of(vec![Col::Nodes(
        buf[start..].iter().map(|&(_, _, n)| n).collect(),
    )]))
}

/// LATE MATERIALIZATION for a sorted `LIMIT` over a `Project`: when the window is
/// a strict PREFIX of the rows (`skip+limit < n`) and every sort key is an output
/// alias (`Slot(i)` into the projection), evaluate ONLY the sort-key expressions
/// over the projection's input to find the top-K rows, then project the FULL item
/// list for just those K survivors — so the non-key columns (a `name` string per
/// row, say) are built for K rows, not all N. `Ok(None)` when the shape doesn't
/// fit (no limit, input not a Project, a key that isn't a projected alias, or the
/// window is the whole set so there is nothing to save).
fn try_late_materialize(
    input: &Plan,
    keys: &[crate::ir::SortKey],
    skip: Option<usize>,
    limit: Option<usize>,
    store: &Store,
    track: bool,
) -> Result<Option<Batch>, String> {
    let Some(limit) = limit else { return Ok(None) };
    if keys.is_empty() {
        return Ok(None);
    }
    let Plan::Project {
        input: pinput,
        items,
    } = input
    else {
        return Ok(None);
    };
    // Every sort key must be an output alias `Slot(i)` — map it to that item's
    // expression, so it can be evaluated over the projection's INPUT.
    let key_exprs: Option<Vec<&Expr>> = keys
        .iter()
        .map(|k| match &k.expr {
            Expr::Slot(i) => items.get(*i).map(|(_, e)| e),
            _ => None,
        })
        .collect();
    let Some(key_exprs) = key_exprs else {
        return Ok(None);
    };

    let base = pull(pinput, store, track)?;
    let n = base.rows();
    let start = skip.unwrap_or(0).min(n);
    let end = start.saturating_add(limit).min(n);
    if end >= n {
        // The window is the whole set — a full projection is unavoidable; nothing
        // to late-materialize.
        return Ok(None);
    }
    if end <= start {
        return Ok(Some(Batch::of(
            items.iter().map(|_| Col::Nodes(vec![])).collect(),
        )));
    }

    // Sort by the key columns evaluated over the base, take the window's rows.
    let key_cols: Vec<Col> = key_exprs
        .iter()
        .map(|e| eval(e, store, &base))
        .collect::<Result<_, _>>()?;
    let key_cols = typed_key_cols(key_cols);
    let mut idx: Vec<usize> = (0..n).collect();
    sort_idx(&mut idx, &key_cols, keys, end);
    let sub = base.gather(&idx[start..end]);

    // NOW project every item, but only over the K surviving rows.
    let cols = eval_all(items.iter().map(|(_, e)| e), store, &sub)?;
    let mut out = Batch::of(cols);
    out.lineage = sub.lineage;
    Ok(Some(out))
}

/// Sort the batch by `keys`, then keep the window `[skip, skip+limit)`. Reorders
/// every slot together, so bound variables stay row-aligned.
fn order_page(
    batch: &Batch,
    store: &Store,
    keys: &[crate::ir::SortKey],
    skip: Option<usize>,
    limit: Option<usize>,
) -> Result<Batch, String> {
    let n = batch.rows();
    let start = skip.unwrap_or(0).min(n);
    let end = limit.map_or(n, |l| start.saturating_add(l).min(n));
    if end <= start {
        return Ok(batch.gather(&[]));
    }
    let mut idx: Vec<usize> = (0..n).collect();
    if !keys.is_empty() {
        let key_cols: Vec<Col> = eval_all(keys.iter().map(|k| &k.expr), store, batch)?;
        let key_cols = typed_key_cols(key_cols);
        sort_idx(&mut idx, &key_cols, keys, end);
    }
    Ok(batch.gather(&idx[start..end]))
}

/// Fold a homogeneous computed key column (`Col::Gen`) into a typed one so `sort_idx`'s
/// raw-f64 / `Arc<str>` arms fire — applied ONLY at a sort, so a plain computed projection
/// keeps the cheap boxed path and does not pay the fold for nothing.
fn typed_key_cols(cols: Vec<Col>) -> Vec<Col> {
    cols.into_iter()
        .map(|c| {
            if let Col::Gen(v) = c {
                typed_col_from_values(v)
            } else {
                c
            }
        })
        .collect()
}

/// Group `batch` by `keys` and compute `aggs` per group. Output slots are the key
/// columns (one value per group, taken from each group's first row) followed by
/// the aggregate columns. Group order is first-seen: a group's index is the
/// order its first row arrived, which is the order it is emitted.
///
/// Rows are labelled with a dense group id in a single pass ([`assign_groups`]),
/// then each aggregate is a single streaming pass over that labelling — so an
/// aggregate never materializes its group's rows, and `count(*)` is a tally, not
/// a bucketed list of row indices.
/// Frontier size below which the storage-direct fold/group loses to eval-then-fold on a
/// compact column (the random `store.column[node]` reads only pay off at scale).
const FRONTIER_FOLD_MIN: usize = 50_000;

/// First-seen grouping for a SINGLE node-PROPERTY key over a materialized batch, via the
/// typed [`frontier_group_by`] (Str/Num/Bool/Dict read off storage) — the general-batch
/// analogue of [`try_dict_grouping`]. Unlike the chain-only `try_frontier_group_fold`, it
/// works on whatever batch the aggregate already pulled (a join, a filtered frontier), so a
/// grouped aggregate over ANY shape skips the `Col::Str` + byte-key `assign_groups` pays.
/// `None` unless the sole key is `Prop{slot, <plain col>}` over a `Col::Nodes` slot backed
/// by a Str/Num/Bool/Dict column.
fn try_node_prop_grouping(
    keys: &[(String, Expr)],
    store: &Store,
    batch: &Batch,
) -> Option<(Vec<u32>, Col, usize)> {
    // Only worth it on a LARGE frontier, where skipping the key `Col` materialization pays.
    // On a small/filtered batch the random `store.column[node]` reads lose to eval'ing a
    // compact key column then grouping it (measured: filtered grouped-aggs regressed).
    if batch.rows() < FRONTIER_FOLD_MIN {
        return None;
    }
    let [(_, Expr::Prop { slot, key })] = keys else {
        return None;
    };
    if key.contains('.') {
        return None;
    }
    let Col::Nodes(frontier) = batch.slot(*slot) else {
        return None;
    };
    let (group_of, key_out, n_groups) = frontier_group_by(store, key, frontier)?;
    Some((group_of, Col::Gen(key_out), n_groups))
}

fn aggregate(
    batch: &Batch,
    store: &Store,
    keys: &[(String, Expr)],
    aggs: &[Agg],
) -> Result<Batch, String> {
    let n = batch.rows();

    // With no keys the whole input is one group, and a scalar aggregate over EMPTY input
    // still emits that one group (SQL: `count(*)` over nothing is 0, one row). A single
    // dict-column key groups by CODE (first-seen) without decoding + string-hashing every
    // row — same group assignment as `assign_groups` on the decoded strings (a code and
    // its string share first-occurrence), so identical groups, order, and per-group
    // summation. Only a dict key over a node frontier takes it; every other key evals its
    // columns and groups as before.
    let (group_of, _first_row, n_groups, key_out) =
        if let Some((g, fr, kc)) = try_dict_grouping(keys, store, batch) {
            let ng = fr.len();
            (g, fr, ng, vec![kc])
        } else if let Some((g, kc, ng)) = try_node_prop_grouping(keys, store, batch) {
            // A single node PROPERTY key (Str/Num/Bool) grouped with the TYPED set read straight
            // off storage — no full `Col::Str` of `Arc` clones + byte key that eval + assign_
            // groups would build. Works on ANY materialized batch (a join, a filtered frontier,
            // a chain), so a grouped aggregate over a comma-join takes it too. Byte-identical.
            (g, Vec::new(), ng, vec![kc])
        } else if keys.is_empty() {
            (vec![0u32; n], Vec::new(), 1, Vec::new())
        } else {
            let key_cols: Vec<Col> = eval_all(keys.iter().map(|(_, e)| e), store, batch)?;
            let (g, fr) = assign_groups(&key_cols, n);
            let ng = fr.len();
            let ko = key_cols.iter().map(|c| c.gather(&fr)).collect();
            (g, fr, ng, ko)
        };

    let mut slots: Vec<Col> = key_out;

    for agg in aggs {
        // Raw min/max over a NUMERIC node-property arg: fold the `f64` off the column with
        // `cmp_num_total`, skipping the `Col::Num` eval and the per-cell `Value` boxing
        // `fold_grouped` pays. Byte-identical (same total order, null skip, all-null → NULL).
        // Only on a large frontier — on a small/filtered batch the random reads lose to
        // eval-then-fold on the compact column.
        if matches!(agg.func, AggFn::Min | AggFn::Max) && n >= FRONTIER_FOLD_MIN {
            if let Some(Expr::Prop { slot, key }) = &agg.arg {
                if let (Col::Nodes(frontier), Some(Column::Num { data, present, .. })) =
                    (batch.slot(*slot), store.column(key))
                {
                    slots.push(Col::Gen(fold_num_minmax(
                        frontier,
                        &group_of,
                        n_groups,
                        data,
                        present,
                        agg.func == AggFn::Min,
                    )));
                    continue;
                }
            }
        }
        let arg_col = agg
            .arg
            .as_ref()
            .map(|e| eval(e, store, batch))
            .transpose()?;
        // Ungrouped fold (`g.V()…fold()` / GQL `collect(x)`) over a VALUE column: the
        // whole input is one list, so CONSUME the column and MOVE each cell into it — no
        // per-element clone the grouped render path pays. Big for a fold of strings
        // (numbers were already cheap Copy). Elements (Nodes/Edges) still take the render
        // path — they must build an element map either way, so moving buys nothing there.
        // Byte-identical: row order and rendering are unchanged.
        let movable = keys.is_empty()
            && !agg.distinct // DISTINCT must dedup — fall through to fold_grouped
            && matches!(agg.func, AggFn::Collect | AggFn::CollectList)
            && matches!(
                arg_col,
                Some(Col::Str(_) | Col::Num(_) | Col::Bool(_) | Col::Gen(_) | Col::Nodes(_))
            );
        if movable {
            let mut list = col_into_values(arg_col.unwrap(), store);
            if agg.func == AggFn::CollectList {
                list.retain(|v| !v.is_null()); // collect_list drops NULLs (Collect keeps)
            }
            slots.push(Col::Gen(vec![Value::List(list)]));
            continue;
        }
        slots.push(Col::Gen(fold_grouped(
            agg,
            arg_col.as_ref(),
            &group_of,
            n_groups,
            store,
        )?));
    }

    Ok(Batch::of(slots))
}

/// Group a node frontier by ONE of its properties, in first-seen order, reading the key
/// straight off storage with a TYPED set — `FnvMap<&str>` for Str (no `Arc` clone, no
/// `Col::Str` materialization), a per-code `Vec` for Dict, `FnvMap<u64>` group-bits for Num,
/// a 2-slot table for Bool. Returns `(group_of, key_out, n_groups)`, byte-identical to
/// `assign_groups`/`try_dict_grouping` over the same key (a value and its code/hash share a
/// first occurrence, absence is one Null group). This is the expensive half of a grouped
/// aggregate over a big frontier — the general path pays a full `Col::Str` of `Arc` clones
/// plus a byte key here; this pays neither.
fn frontier_group_by(
    store: &Store,
    key: &str,
    frontier: &[u32],
) -> Option<(Vec<u32>, Vec<Value>, usize)> {
    let col = store.column(key)?;
    let mut group_of: Vec<u32> = Vec::with_capacity(frontier.len());
    let mut key_out: Vec<Value> = Vec::new();
    let mut null_group: Option<u32> = None;
    macro_rules! null_g {
        () => {{
            *null_group.get_or_insert_with(|| {
                let g = key_out.len() as u32;
                key_out.push(Value::Null);
                g
            })
        }};
    }
    match col {
        Column::Str { data, present, .. } => {
            let mut seen: FnvMap<&str, u32> = FnvMap::default();
            for &node in frontier {
                let g = if node != u32::MAX && present[node as usize] {
                    let s = data[node as usize].as_ref();
                    if let Some(&g) = seen.get(s) {
                        g
                    } else {
                        let g = key_out.len() as u32;
                        seen.insert(s, g);
                        key_out.push(Value::Str(data[node as usize].clone()));
                        g
                    }
                } else {
                    null_g!()
                };
                group_of.push(g);
            }
        }
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            let mut code_to_group: Vec<u32> = vec![u32::MAX; dict.len()];
            for &node in frontier {
                let g = if node != u32::MAX && present[node as usize] {
                    let c = codes[node as usize] as usize;
                    if code_to_group[c] == u32::MAX {
                        code_to_group[c] = key_out.len() as u32;
                        key_out.push(Value::Str(dict[c].clone()));
                    }
                    code_to_group[c]
                } else {
                    null_g!()
                };
                group_of.push(g);
            }
        }
        Column::Num { data, present, .. } => {
            let mut seen: FnvMap<u64, u32> = FnvMap::default();
            for &node in frontier {
                let g = if node != u32::MAX && present[node as usize] {
                    let bits = value::num_group_bits(data[node as usize]);
                    if let Some(&g) = seen.get(&bits) {
                        g
                    } else {
                        let g = key_out.len() as u32;
                        seen.insert(bits, g);
                        key_out.push(Value::Num(data[node as usize]));
                        g
                    }
                } else {
                    null_g!()
                };
                group_of.push(g);
            }
        }
        Column::Bool { data, present, .. } => {
            let mut slot: [Option<u32>; 2] = [None, None];
            for &node in frontier {
                let g = if node != u32::MAX && present[node as usize] {
                    let b = usize::from(data[node as usize]);
                    *slot[b].get_or_insert_with(|| {
                        let g = key_out.len() as u32;
                        key_out.push(Value::Bool(data[node as usize]));
                        g
                    })
                } else {
                    null_g!()
                };
                group_of.push(g);
            }
        }
        _ => return None, // Temporal / Gen → the general path
    }
    let n = key_out.len();
    Some((group_of, key_out, n))
}

/// Fused grouped aggregate over a large hop-chain frontier: build `group_of` with the typed
/// [`frontier_group_by`] (skipping the full key `Col::Str` + byte key the general
/// [`aggregate`] pays) and reuse [`fold_grouped`] for the aggregates — byte-identical, but
/// without materializing the exploded frontier's KEY column. Wins exactly the case the
/// diagnostics isolated: a high-cardinality string group key over a multi-hop frontier.
/// `None` unless the key and every agg arg are plain properties of the chain frontier.
fn try_frontier_group_fold(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    let [(
        _,
        Expr::Prop {
            slot: kslot,
            key: kkey,
        },
    )] = keys
    else {
        return None;
    };
    if kkey.contains('.') {
        return None;
    }
    let width = chain_width(input).or_else(|| chain_pull_width(input))?;
    if *kslot != width - 1 {
        return None;
    }
    // Every agg arg must be a plain frontier property (or none, for count(*)) — so it reads
    // off the single-slot frontier batch after a slot remap. Any richer arg → general path.
    for a in aggs {
        match &a.arg {
            None => {}
            Some(Expr::Prop { slot, key }) if *slot == width - 1 && !key.contains('.') => {}
            _ => return None,
        }
    }
    // Gate on the ESTIMATED frontier size BEFORE any traversal/pull, so a SELECTIVE filter
    // (a small frontier) bails here with ZERO wasted work. Pulling and then bailing on a
    // post-hoc size check would double the pull for the fallback — a measured 0.30x
    // regression. The estimator models filter selectivity (an indexed `=` is exact); a wrong
    // estimate only costs time (byte-identical routes), never a row.
    const FUSED_GROUP_ROWS: f64 = 100_000.0;
    if crate::cost::estimate(input, store).rows < FUSED_GROUP_ROWS {
        return None;
    }
    // A pure Scan/Expand chain gets its frontier cheaply and directly; a FILTERED chain is
    // pulled ONCE (the pull applies every filter) and its endpoint slot IS the filtered
    // frontier — same rows, same order the general path would group.
    let frontier = match frontier_ids(input, store) {
        Some(f) => f,
        None => {
            let b = pull(input, store, false).ok()?;
            let Col::Nodes(f) = b.slot(width - 1) else {
                return None;
            };
            f.clone()
        }
    };
    let (group_of, key_out, n_groups) = frontier_group_by(store, kkey, &frontier)?;
    let mut slots: Vec<Col> = vec![Col::Gen(key_out)];
    let mut fb: Option<Batch> = None; // built lazily (only for args that need fold_grouped)
    for agg in aggs {
        // min/max over a NUMERIC frontier property folds RAW off the column (`f64`,
        // `cmp_num_total`) — no per-row `value_at` boxing to `Value` + general `cmp_total`,
        // which the diagnostics showed is the whole remaining cost over a big frontier.
        if matches!(agg.func, AggFn::Min | AggFn::Max) {
            if let Some(Expr::Prop { key, .. }) = &agg.arg {
                if let Some(Column::Num { data, present, .. }) = store.column(key) {
                    slots.push(Col::Gen(fold_num_minmax(
                        &frontier,
                        &group_of,
                        n_groups,
                        data,
                        present,
                        agg.func == AggFn::Min,
                    )));
                    continue;
                }
            }
        }
        // count(*) needs no arg column; anything else reads the arg off the frontier (slot 0
        // of `fb`) and reuses the general byte-identical fold.
        let arg_col = match &agg.arg {
            None => None,
            Some(Expr::Prop { key, .. }) => {
                let b = fb.get_or_insert_with(|| Batch::single(Col::Nodes(frontier.clone())));
                Some(
                    eval(
                        &Expr::Prop {
                            slot: 0,
                            key: key.clone(),
                        },
                        store,
                        b,
                    )
                    .ok()?,
                )
            }
            _ => return None,
        };
        slots.push(Col::Gen(
            fold_grouped(agg, arg_col.as_ref(), &group_of, n_groups, store).ok()?,
        ));
    }
    Some(Batch::of(slots))
}

/// Per-group min/max of a NUMERIC frontier column, folded RAW: `f64` compared with
/// `cmp_num_total` (the same total order [`fold_grouped`]'s `cmp_total` uses for two
/// numbers), a NULL/absent cell skipped, an all-null group → NULL. Avoids boxing every cell
/// to a `Value` and the general comparison — byte-identical, far cheaper over a big frontier.
fn fold_num_minmax(
    frontier: &[u32],
    group_of: &[u32],
    n_groups: usize,
    data: &[f64],
    present: &[bool],
    want_min: bool,
) -> Vec<Value> {
    let mut best: Vec<Option<f64>> = vec![None; n_groups];
    for (i, &g) in group_of.iter().enumerate() {
        let node = frontier[i];
        if node == u32::MAX || !present[node as usize] {
            continue;
        }
        let x = data[node as usize];
        let slot = &mut best[g as usize];
        *slot = Some(match *slot {
            None => x,
            Some(cur) => {
                let ord = value::cmp_num_total(x, cur);
                if (want_min && ord.is_lt()) || (!want_min && ord.is_gt()) {
                    x
                } else {
                    cur
                }
            }
        });
    }
    best.into_iter()
        .map(|o| o.map_or(Value::Null, Value::Num))
        .collect()
}

/// First-seen grouping for a SINGLE dict-column key, by code — avoiding the per-row dict
/// decode + string hash `assign_groups` would pay. Returns `(group_of, first_row,
/// key_output_col)`; the key output is each group's decoded value (absent → NULL), built
/// once per group. `None` unless the sole key is `Prop{slot, <dict col>}` over a node
/// frontier. Byte-identical grouping: a code and its string share their first occurrence.
fn try_dict_grouping(
    keys: &[(String, Expr)],
    store: &Store,
    batch: &Batch,
) -> Option<(Vec<u32>, Vec<usize>, Col)> {
    let [(_, Expr::Prop { slot, key })] = keys else {
        return None;
    };
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(key)
    else {
        return None;
    };
    let Col::Nodes(ids) = batch.slot(*slot) else {
        return None;
    };
    let mut code_to_group: Vec<u32> = vec![u32::MAX; dict.len()];
    let mut null_group: Option<u32> = None;
    let mut first_row: Vec<usize> = Vec::new();
    let mut key_vals: Vec<Value> = Vec::new();
    let mut group_of: Vec<u32> = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let g = if id != u32::MAX && present[id as usize] {
            let c = codes[id as usize] as usize;
            if code_to_group[c] == u32::MAX {
                code_to_group[c] = first_row.len() as u32;
                first_row.push(i);
                key_vals.push(Value::Str(dict[c].clone()));
            }
            code_to_group[c]
        } else {
            *null_group.get_or_insert_with(|| {
                let g = first_row.len() as u32;
                first_row.push(i);
                key_vals.push(Value::Null);
                g
            })
        };
        group_of.push(g);
    }
    Some((group_of, first_row, Col::Gen(key_vals)))
}

/// Assign a dense, first-seen group id to every row from its key columns.
/// Returns `(group_of, first_row)`: `group_of[i]` is row `i`'s group,
/// `first_row[g]` the row that opened group `g`. A single native key column is
/// grouped on its raw type with no boxing; anything else falls back to a reused
/// byte key ([`value::group_key_into`]). Both honor the one grouping contract.
fn assign_groups(key_cols: &[Col], n: usize) -> (Vec<u32>, Vec<usize>) {
    if let [only] = key_cols {
        match only {
            Col::Num(v) => return group_by(n, v.iter().map(|&x| value::num_group_bits(x))),
            // Node ids are small non-negative integers: their id IS the key, and
            // it matches `value_at` (which surfaces a node as `Num(id)`, whose
            // group bits are the same integer).
            Col::Nodes(v) | Col::Edges(v) => return group_by(n, v.iter().map(|&x| u64::from(x))),
            Col::Bool(v) => return group_by(n, v.iter().map(|&b| u64::from(b))),
            Col::Str(v) => return group_by_arc(v),
            Col::Gen(_) => {} // mixed: fall through to the byte-key path
        }
    }
    // General path: self-delimiting byte key per row, reused buffer, allocate
    // only when a row opens a new group.
    let mut of: FnvMap<Vec<u8>, u32> = FnvMap::default();
    let mut group_of = Vec::with_capacity(n);
    let mut first_row = Vec::new();
    let mut buf = Vec::new();
    for i in 0..n {
        buf.clear();
        for kc in key_cols {
            value::group_key_into(&kc.value_at(i), &mut buf);
        }
        let g = match of.get(buf.as_slice()) {
            Some(&g) => g,
            None => {
                let g = first_row.len() as u32;
                of.insert(buf.clone(), g);
                first_row.push(i);
                g
            }
        };
        group_of.push(g);
    }
    (group_of, first_row)
}

/// Group by a per-row key of a `Hash + Eq` type (the typed fast path).
fn group_by<K: std::hash::Hash + Eq>(
    n: usize,
    keys: impl Iterator<Item = K>,
) -> (Vec<u32>, Vec<usize>) {
    let mut of: FnvMap<K, u32> = FnvMap::default();
    let mut group_of = Vec::with_capacity(n);
    let mut first_row = Vec::new();
    for (i, k) in keys.enumerate() {
        let g = match of.get(&k) {
            Some(&g) => g,
            None => {
                let g = first_row.len() as u32;
                of.insert(k, g);
                first_row.push(i);
                g
            }
        };
        group_of.push(g);
    }
    (group_of, first_row)
}

/// Group a string column keyed on the `Arc<str>` itself: a row that opens a new
/// group stores a clone of the shared pointer (a refcount bump), NOT a freshly
/// allocated `Box<str>` — so a million distinct strings cost a million refcount
/// bumps, not a million heap allocations + copies. Lookups borrow `&str`, so a
/// repeated string never touches the allocator.
fn group_by_arc(keys: &[Arc<str>]) -> (Vec<u32>, Vec<usize>) {
    // Pre-size for the worst case (all-distinct) so the map never rehashes while
    // filling — the rehash chain dominated an all-unique million-key merge.
    let mut of: FnvMap<Arc<str>, u32> =
        FnvMap::with_capacity_and_hasher(keys.len(), Default::default());
    let mut group_of = Vec::with_capacity(keys.len());
    let mut first_row = Vec::new();
    for (i, k) in keys.iter().enumerate() {
        let g = match of.get(k.as_ref()) {
            Some(&g) => g,
            None => {
                let g = first_row.len() as u32;
                of.insert(Arc::clone(k), g);
                first_row.push(i);
                g
            }
        };
        group_of.push(g);
    }
    (group_of, first_row)
}

/// Sort one cell in place for `order(local)`: a `List` by its elements, a `Map`
/// by its values (TinkerPop's default local map ordering), anything else
/// unchanged. Order is the value contract's `cmp_total`; `descending` reverses.
fn sort_local_cell(v: Value, descending: bool, by_key: bool) -> Value {
    let dir = |ord: std::cmp::Ordering| if descending { ord.reverse() } else { ord };
    match v {
        Value::List(mut items) => {
            items.sort_by(|a, b| dir(value::cmp_total(a, b)));
            Value::List(items)
        }
        Value::Map(pairs) => {
            let mut pairs = (*pairs).clone();
            // `by(values)` (the default) sorts on the entry value; `by(keys)` on the key.
            pairs.sort_by(|a, b| {
                let (l, r) = if by_key { (&a.0, &b.0) } else { (&a.1, &b.1) };
                dir(value::cmp_total(l, r))
            });
            Value::Map(std::sync::Arc::new(pairs))
        }
        other => other,
    }
}

/// Fold one aggregate to one value per group in a single streaming pass over the
/// group labelling. Null policy and ordering come from the value contract;
/// nothing here restates them.
fn fold_grouped(
    agg: &Agg,
    arg_col: Option<&Col>,
    group_of: &[u32],
    n_groups: usize,
    store: &Store,
) -> Result<Vec<Value>, String> {
    // `count(*)` — no argument — is each group's row count: a pure tally.
    if agg.func == AggFn::Count && agg.arg.is_none() {
        let mut tally = vec![0f64; n_groups];
        for &g in group_of {
            tally[g as usize] += 1.0;
        }
        return Ok(tally.into_iter().map(Value::Num).collect());
    }
    let Some(col) = arg_col else {
        return Ok(vec![Value::Null; n_groups]); // sum/min/max/avg with no argument
    };

    // A DISTINCT aggregate (other than the `Count` arm below, which counts distinct
    // itself) drops duplicate values per group BEFORE folding: a row whose value has
    // already appeared in its group is routed to a throwaway SINK group so every fold
    // arm below simply ignores it — no per-arm change. The sink group's result is
    // truncated off at the end. Fixes `collect_list(DISTINCT …)`, `min(DISTINCT …)`,
    // etc. previously folding over duplicates.
    let orig_groups = n_groups;
    let sink_remap: Option<Vec<u32>> = if agg.distinct && agg.func != AggFn::Count {
        let mut seen: Vec<FnvSet<Vec<u8>>> = (0..n_groups).map(|_| FnvSet::default()).collect();
        let sink = n_groups as u32;
        let mut remapped = Vec::with_capacity(group_of.len());
        for (i, &g) in group_of.iter().enumerate() {
            let mut buf = Vec::new();
            value::group_key_into(&col.value_at(i), &mut buf);
            remapped.push(if seen[g as usize].insert(buf) {
                g
            } else {
                sink
            });
        }
        Some(remapped)
    } else {
        None
    };
    let group_of: &[u32] = sink_remap.as_deref().unwrap_or(group_of);
    let n_groups = if sink_remap.is_some() {
        orig_groups + 1
    } else {
        orig_groups
    };

    let mut out: Vec<Value> = match agg.func {
        AggFn::Count if agg.distinct => {
            // Per-group distinct count. A dedicated set per group, keyed by the
            // grouping bytes; a group entry is allocated only for a new value.
            let mut sets: Vec<FnvSet<Vec<u8>>> = (0..n_groups).map(|_| FnvSet::default()).collect();
            let mut buf = Vec::new();
            for (i, &g) in group_of.iter().enumerate() {
                let v = col.value_at(i);
                if v.is_null() {
                    continue;
                }
                buf.clear();
                value::group_key_into(&v, &mut buf);
                let set = &mut sets[g as usize];
                if !set.contains(buf.as_slice()) {
                    set.insert(buf.clone());
                }
            }
            sets.iter().map(|s| Value::Num(s.len() as f64)).collect()
        }
        AggFn::Count => {
            // count(arg): non-null values per group.
            let mut tally = vec![0f64; n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                if !col.value_at(i).is_null() {
                    tally[g as usize] += 1.0;
                }
            }
            tally.into_iter().map(Value::Num).collect()
        }
        AggFn::Sum | AggFn::Avg => {
            // total + count of non-null NUMERIC values. A non-null NON-numeric value
            // (duration/date/string/list) is a DATA EXCEPTION — sum()/avg() never
            // coerce (the same SQL rule as binary arithmetic). NULLs are skipped. SUM
            // of an empty/all-null group is 0, AVG is NULL (no values to divide).
            let mut total = vec![0f64; n_groups];
            let mut cnt = vec![0u64; n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                match col.value_at(i) {
                    Value::Null => {}
                    Value::Num(x) => {
                        total[g as usize] += x;
                        cnt[g as usize] += 1;
                    }
                    // A Gremlin multi-key `values('v','k')` arg is a LIST — flatten it,
                    // summing each numeric element (skipping non-numeric/null).
                    Value::List(items) if agg.null_on_empty => {
                        for el in &items {
                            if let Value::Num(x) = el {
                                total[g as usize] += x;
                                cnt[g as usize] += 1;
                            }
                        }
                    }
                    // GQL faults on a non-numeric sum/avg; Gremlin (`null_on_empty`
                    // marker) SKIPS it, like a null (so sum of {"text", 4} is 4).
                    _ if agg.null_on_empty => {}
                    _ => return Err("sum()/avg() require numeric values".into()),
                }
            }
            (0..n_groups)
                .map(|g| {
                    if agg.func == AggFn::Sum {
                        if cnt[g] == 0 && agg.null_on_empty {
                            Value::Null // Gremlin sum() of nothing is NULL
                        } else {
                            Value::Num(total[g]) // 0.0 when cnt == 0 (GQL/SQL)
                        }
                    } else if cnt[g] == 0 {
                        Value::Null // AVG of nothing
                    } else {
                        Value::Num(total[g] / cnt[g] as f64)
                    }
                })
                .collect()
        }
        AggFn::Collect | AggFn::CollectList => {
            // Gather each group's values into a list, in row order (a preceding sort
            // carries through). `Collect` (Gremlin fold) KEEPS nulls; `CollectList`
            // (GQL collect_list) SKIPS them, matching core. An empty (or all-null,
            // for CollectList) group folds to the empty list.
            let skip_nulls = agg.func == AggFn::CollectList;
            let mut lists: Vec<Vec<Value>> = vec![Vec::new(); n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                // A folded VERTEX/EDGE renders as its element map (same as a top-level
                // one), not the raw dense id — so a `fold()`/`aggregate` of elements is
                // self-describing and canonicalizes like `g.V()` does.
                let v = render_cell(col, i, store);
                if skip_nulls && v.is_null() {
                    continue;
                }
                lists[g as usize].push(v);
            }
            lists.into_iter().map(Value::List).collect()
        }
        AggFn::Min | AggFn::Max => {
            let want_min = agg.func == AggFn::Min;
            let mut best: Vec<Option<Value>> = vec![None; n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                let v = col.value_at(i);
                if v.is_null() {
                    continue;
                }
                match &best[g as usize] {
                    None => best[g as usize] = Some(v),
                    Some(cur) => {
                        let ord = value::cmp_total(&v, cur);
                        if (want_min && ord.is_lt()) || (!want_min && ord.is_gt()) {
                            best[g as usize] = Some(v);
                        }
                    }
                }
            }
            best.into_iter().map(|o| o.unwrap_or(Value::Null)).collect()
        }
        AggFn::StddevPop | AggFn::StddevSamp => {
            // One-pass moments per group: a present non-null value contributes as a
            // number (a non-numeric one as NaN, which propagates — matching core's
            // stddev over a non-numeric column). NULLs are skipped.
            let sample = agg.func == AggFn::StddevSamp;
            let mut sum = vec![0f64; n_groups];
            let mut sum_sq = vec![0f64; n_groups];
            let mut cnt = vec![0u64; n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                let x = match col.value_at(i) {
                    Value::Null => continue,
                    Value::Num(x) => x,
                    // A non-null non-numeric value is a data exception — stddev never
                    // coerces, matching sum()/avg() and the numeric scalar functions.
                    _ => return Err("stddev() requires numeric values".into()),
                };
                let g = g as usize;
                sum[g] += x;
                sum_sq[g] += x * x;
                cnt[g] += 1;
            }
            (0..n_groups)
                .map(|g| stddev_of(cnt[g], sum[g], sum_sq[g], sample))
                .collect()
        }
        AggFn::PercentileCont | AggFn::PercentileDisc => {
            // Ordered-set: gather each group's finite numeric values, sort, and take
            // the `frac`-th percentile (interpolated for cont, discrete for disc) —
            // replicated from core's `percentile`. Empty group → NULL.
            let cont = agg.func == AggFn::PercentileCont;
            let frac = agg.frac.unwrap_or(0.0);
            let mut per_group: Vec<Vec<f64>> = vec![Vec::new(); n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                match col.value_at(i) {
                    Value::Null => {}
                    // A non-null non-numeric value is a data exception — percentile never
                    // coerces, matching sum()/avg() and the numeric scalar functions.
                    Value::Num(x) if x.is_finite() => per_group[g as usize].push(x),
                    Value::Num(_) => {}
                    _ => return Err("percentile() requires numeric values".into()),
                }
            }
            per_group
                .into_iter()
                .map(|nums| percentile_of(nums, frac, cont))
                .collect()
        }
    };
    out.truncate(orig_groups); // drop the DISTINCT sink group (no-op otherwise)
    Ok(out)
}

/// The `frac`-th percentile of `nums` — interpolated (`cont`) or discrete (`disc`) —
/// replicated exactly from core's `percentile`. Empty input → NULL.
fn percentile_of(mut nums: Vec<f64>, frac: f64, cont: bool) -> Value {
    if nums.is_empty() {
        return Value::Null;
    }
    nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = nums.len();
    let result = if cont {
        let rn = frac * (n - 1) as f64;
        let lo = rn.floor() as usize;
        let hi = rn.ceil() as usize;
        if lo == hi {
            nums[lo]
        } else {
            nums[lo] + (rn - lo as f64) * (nums[hi] - nums[lo])
        }
    } else {
        let idx = ((frac * n as f64).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        nums[idx]
    };
    Value::Num(result)
}

/// Population / sample standard deviation from one-pass moments — replicated exactly
/// from core's `stddev_of`. `pop` is NULL over 0 rows, `samp` over fewer than 2; the
/// summed squared deviation is clamped at 0 (preserving NaN) so f64 cancellation
/// can't slip a tiny negative into `sqrt`.
fn stddev_of(n: u64, sum: f64, sum_sq: f64, sample: bool) -> Value {
    let denom = if sample {
        if n < 2 {
            return Value::Null;
        }
        (n - 1) as f64
    } else {
        if n == 0 {
            return Value::Null;
        }
        n as f64
    };
    let nf = n as f64;
    let variance = (sum_sq - sum * sum / nf) / denom;
    let clamped = if variance.is_nan() {
        f64::NAN
    } else {
        variance.max(0.0)
    };
    Value::Num(clamped.sqrt())
}

/// A hop: for each input row, expand the node in slot `from` along `dir`,
/// filtered by `edge_label`; emit one output row per matching neighbour with the
/// existing slots replicated and the neighbour appended as a new slot. This is
/// the bulk (lineage-free) strategy: `keep` records which input row each output
/// row came from, `nbrs` the landed node — the existing slots are gathered by
fn reverse_dir(dir: Dir) -> Dir {
    match dir {
        Dir::Out => Dir::In,
        Dir::In => Dir::Out,
        Dir::Both => Dir::Both,
    }
}

/// Cardinality-driven ANCHOR FLIP for `Filter(target = lit, Expand(Scan(src), Out))`
/// — the "selective filter on the traversal TARGET" shape. The forward plan scans
/// EVERY source and expands to filter the target at the end; when the target is an
/// indexed `=` whose bucket is smaller than the source scan, it is far cheaper to
/// SEED the target (index) and walk the edges in REVERSE to the sources. The output
/// is the IDENTICAL `[source, target]` slot layout, so nothing downstream changes;
/// only the (unspecified) row order differs — the multiset is preserved. `None`
/// unless the shape matches, the target is index-seekable, the cost says flip, and
/// no path is tracked / no edge is bound.
/// The reverse-walk win for shortest paths: `Filter{endpoint == t} over ShortestPath`
/// resolves the target set `t` and hands it to the BFS as an early stop, so each source
/// stops sweeping the moment every target is settled rather than exploring the whole
/// reachable component. The filter is STILL applied to the result here, so the output is
/// byte-identical to the unbounded path — this only avoids materializing rows the filter
/// would drop. Conservative: only an indexed `endpoint.key == lit` over `min == 0`.
fn try_shortest_early_stop(pred: &Expr, input: &Plan, store: &Store, track: bool) -> Option<Batch> {
    let Plan::ShortestPath {
        input: sp_in,
        from,
        dir,
        edge_label,
        min: 0, // `*` only — a `+` (min 1) has source-as-endpoint cycle cases
        max,
        selector,
        edge_pred,
    } = input
    else {
        return None;
    };
    let (key, value) = target_eq(pred, 1)?; // endpoint (slot 1) `== lit`
    if !store.has_hash_index(&key) {
        return None; // resolve the target set from the index, else keep the normal path
    }
    let targets = store.index_lookup(&key, &value)?;
    if targets.is_empty() {
        return None; // no target → the normal path yields empty; nothing to accelerate
    }
    let sp_batch = shortest_path(
        &pull(sp_in, store, track).ok()?,
        store,
        *from,
        *dir,
        edge_label,
        0,
        *max,
        *selector,
        edge_pred.as_deref(),
        Some(&targets),
    );
    // Apply the endpoint filter exactly as the general path would — the early stop only
    // changed which never-kept rows were produced, so this reproduces the same result.
    let keep: Vec<usize> = match try_filter_keep(pred, store, &sp_batch) {
        Some(k) => k,
        None => {
            let mask = eval(pred, store, &sp_batch).ok()?;
            match &mask {
                Col::Bool(bs) => (0..bs.len()).filter(|&i| bs[i]).collect(),
                other => (0..other.len())
                    .filter(|&i| other.value_at(i).is_true())
                    .collect(),
            }
        }
    };
    Some(sp_batch.gather(&keep))
}

/// The cardinality-approved decision for a reverse-seed: the hop chain (innermost-first,
/// carrying each hop's bind-edge flag), the source scan's label, the endpoint slot the
/// predicate seeds on, and the seeded endpoint bucket. Produced by [`reverse_seed_decide`],
/// which materializes the bucket to size the cardinality guard.
struct RevSeed {
    hops: Vec<RevHop>,
    src_label: Option<String>,
    ep_slot: usize,
    bucket: Vec<u32>,
}

fn reverse_seed_decide(pred: &Expr, input: &Plan, store: &Store, track: bool) -> Option<RevSeed> {
    if track {
        return None; // a path-reading query keeps the forward walk (lineage)
    }
    // Unwrap a chain of expands over a scan. A hop may bind its edge (appending an edge
    // slot before the landed node); `chain` collects them outermost-in as
    // (from, dir, edge_label, bind_edge).
    let mut chain: Vec<(usize, Dir, &[String], bool)> = Vec::new();
    let mut cur = input;
    let src_label = loop {
        match cur {
            Plan::Expand {
                input: inner,
                from,
                dir,
                edge_label,
                bind_edge,
                double_loops: false,
            } => {
                chain.push((*from, *dir, edge_label.as_slice(), *bind_edge));
                cur = inner.as_ref();
            }
            Plan::Scan { label } if !chain.is_empty() => break label.clone(),
            _ => return None, // source must bottom at an unfiltered scan
        }
    };
    // Build the hops innermost-first, verifying each feeds from the running node slot (a
    // straight chain, no branch/re-entry) and tracking where the endpoint node lands: a
    // bound hop appends an edge slot then the node (+2), an unbound hop just the node (+1).
    let mut hops: Vec<RevHop> = Vec::with_capacity(chain.len());
    let mut node_slot = 0usize;
    for &(from, dir, edge_label, bind_edge) in chain.iter().rev() {
        if from != node_slot {
            return None;
        }
        node_slot += if bind_edge { 2 } else { 1 };
        let want = match want_etypes(store, edge_label) {
            Ok(w) => w,
            Err(()) => return None,
        };
        hops.push(RevHop {
            dir,
            want,
            bind_edge,
        });
    }
    let ep_slot = node_slot;
    // Seed the endpoint from the index — an equality, range, IN, OR, or the more selective
    // conjunct of an AND; the residual filter (below) exacts the answer.
    let bucket = seed_bucket(pred, ep_slot, store)?;
    // Cardinality decision: flip only when the endpoint bucket is smaller than the source
    // scan (the reverse walks back only the paths that reach it).
    let source_rows = match &src_label {
        Some(l) => store.nodes_with_label(l).len(),
        None => store.live_node_count(),
    };
    // Loose only for a bare equality (no residual, and a smaller-than-scan bucket already
    // wins). Everything else — range / IN / OR / AND — MATERIALIZES the walked rows and
    // boxes a residual, so it needs the selectivity guard; a large OR union in particular
    // must NOT flip when a downstream LIMIT could stream the forward walk cheaply.
    let loose = target_eq(pred, ep_slot).is_some();
    // A SINGLE-hop non-loose seed has a cheap forward alternative: sweep the endpoint type's
    // edges off the per-type CSR (`fwd`). Reverse-seeding a NON-selective range instead seeds
    // its large bucket and walks the SPARSE type-in edges back — a random probe per seed that
    // costs more than the sequential forward sweep. Decline when the bucket is not smaller than
    // that forward cost (the `reverse_seed_worth` guard prices against the SOURCE scan, which
    // over-fires for a sparse type — `age >= 77` = 92k seeds vs 80k forward F edges). Byte-
    // identical either way — this only picks the cheaper equivalent plan.
    if !loose {
        if let Some(ep) = hops.last() {
            let fwd: usize = ep
                .want
                .iter()
                .filter_map(|&t| store.out_typed_flat(t).map(<[_]>::len))
                .sum();
            // A random type-in probe per seed costs ~10x a sequential forward edge read, so a
            // SINGLE-hop reverse-seed wins only when its bucket is well under a tenth of `fwd`
            // (the endpoint type's forward edge count). A MULTI-hop forward walk fans out again
            // past `fwd`, so it is only cheaper when the bucket is at least the whole endpoint
            // edge count — a stricter bar that keeps a mid-selective 2-hop range on the seed.
            let factor = if hops.len() == 1 { 8 } else { 1 };
            if fwd > 0 && bucket.len().saturating_mul(factor) >= fwd {
                return None;
            }
        }
    }
    if !reverse_seed_worth(bucket.len(), source_rows, loose, store) {
        return None;
    }
    Some(RevSeed {
        hops,
        src_label,
        ep_slot,
        bucket,
    })
}

/// A `DISTINCT` whose result cannot reach `cap` rows: its input projects a single bare
/// property that is a low-cardinality dict column with `distinct_count + 1 <= cap` (the
/// `+1` covers a possible NULL, which DISTINCT counts as a value but the dict does not).
/// A `LIMIT cap` over such a DISTINCT cannot bind.
fn distinct_cap_cannot_bind(distinct_input: &Plan, cap: usize, store: &Store) -> bool {
    let Plan::Project { items, .. } = distinct_input else {
        return false;
    };
    if items.len() != 1 {
        return false;
    }
    let Expr::Prop { key, .. } = &items[0].1 else {
        return false;
    };
    store
        .distinct_count(key)
        .is_some_and(|d| d.saturating_add(1) <= cap)
}

/// Would `plan` reverse-seed? Peeks through the row-preserving wrappers that sit above
/// the `Filter` (Project) so a blocking op (OrderPage) can pick the reverse-seed over a
/// forward stream. Cheap — the underlying decision is O(1) index lookups, no walk.
fn reverse_seed_applies(plan: &Plan, store: &Store, track: bool) -> bool {
    match plan {
        Plan::Project { input, .. } => reverse_seed_applies(input, store, track),
        Plan::Filter { input, pred } => reverse_seed_decide(pred, input, store, track).is_some(),
        _ => false,
    }
}

/// A reverse-walk hop: direction, edge-type want-set, and whether the forward hop BOUND
/// its edge (appending an edge slot before the landed node). Innermost-first.
struct RevHop {
    dir: Dir,
    want: Vec<u32>,
    bind_edge: bool,
}

/// Reverse-walk a chain of hops, prepending each hop's source (and, for a bound-edge hop,
/// its edge) to every partial row. `rows` start as suffixes headed by the frontier node
/// (`row[0]`, the node to walk back from); a bound hop prepends `[src, edge]` so the row
/// stays in forward slot order `[…, src, edge, landed, …]` and `row[0]` remains a node.
/// Intermediate nodes are unconstrained (the forward plan filters only the scan's source),
/// so the scan's source label is enforced once, on the last prepend (hop 0 → s_0). An
/// empty `hops` returns `rows` unchanged.
fn reverse_walk_chain(
    mut rows: Vec<Vec<u32>>,
    hops: &[RevHop],
    src_label: Option<&str>,
    store: &Store,
) -> Vec<Vec<u32>> {
    for k in (0..hops.len()).rev() {
        let hop = &hops[k];
        let rev = reverse_dir(hop.dir);
        let last_hop = k == 0;
        let mut next: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        for row in &rows {
            let head = row[0];
            for_each_nbr(store, head, rev, &hop.want, false, |a, eid| {
                if last_hop && !src_label.is_none_or(|l| store.is_labeled(a, l)) {
                    return;
                }
                let mut r = Vec::with_capacity(row.len() + if hop.bind_edge { 2 } else { 1 });
                r.push(a);
                if hop.bind_edge {
                    r.push(eid);
                }
                r.extend_from_slice(row);
                next.push(r);
            });
        }
        rows = next;
    }
    rows
}

/// The per-slot column kinds a reverse-walk of `hops` produces, in forward slot order:
/// the source node, then each hop's `[edge?, landed node]`. `false` = node, `true` = edge.
fn chain_slot_kinds(hops: &[RevHop]) -> Vec<bool> {
    let mut kinds = vec![false]; // slot 0: the source node
    for hop in hops {
        if hop.bind_edge {
            kinds.push(true); // the bound edge slot
        }
        kinds.push(false); // the landed node
    }
    kinds
}

/// Transpose full rows into columns typed by `kinds` (`true` = edge slot, else node).
fn rows_to_batch(rows: &[Vec<u32>], kinds: &[bool]) -> Batch {
    let mut cols: Vec<Vec<u32>> = (0..kinds.len())
        .map(|_| Vec::with_capacity(rows.len()))
        .collect();
    for row in rows {
        for (i, &v) in row.iter().enumerate() {
            cols[i].push(v);
        }
    }
    Batch::of(
        cols.into_iter()
            .zip(kinds)
            .map(|(c, &is_edge)| {
                if is_edge {
                    Col::Edges(c)
                } else {
                    Col::Nodes(c)
                }
            })
            .collect(),
    )
}

fn try_reverse_expand(pred: &Expr, input: &Plan, store: &Store, track: bool) -> Option<Batch> {
    let RevSeed {
        hops,
        src_label,
        ep_slot,
        bucket,
    } = reverse_seed_decide(pred, input, store, track)?;
    // Reverse-walk the chain from the seeded endpoint bucket to the labeled sources, then
    // transpose into node/edge columns matching the forward slot layout.
    let rows = reverse_walk_chain(
        bucket.iter().map(|&t| vec![t]).collect(),
        &hops,
        src_label.as_deref(),
        store,
    );
    let b = rows_to_batch(&rows, &chain_slot_kinds(&hops));
    // A bare equality is fully satisfied by the seed. A conjunction / range / IN / OR — or
    // a bound-edge chain with an edge-property residual — needs the WHOLE predicate applied
    // over the (small) seeded batch. If the residual can't evaluate cleanly, decline.
    if target_eq(pred, ep_slot).is_some() {
        return Some(b);
    }
    let keep = residual_keep(pred, store, &b)?;
    Some(b.gather(&keep))
}

/// The reverse-seed for a VAR-LENGTH hop, possibly behind fixed leading hops:
/// `Filter(endpoint.key = lit, VarLength{from:L}(Expand·…·Expand(Scan)))` with L ≥ 0
/// plain fixed hops before the quantifier. The forward walk enumerates every path from
/// every source and filters the endpoint at the end — on a fanning graph that is the
/// ~34M-path materialization that trips the trail-limit guard. Instead, seed the
/// selective indexed endpoint and walk BACKWARD: the var-length in reverse (reusing
/// `var_length` on `reverse_dir(dir)`), then the fixed chain in reverse. Reversal
/// preserves every mode's path validity (Trail = same edge set, Simple/Acyclic = same
/// node set) and forward/reverse trails biject, so the whole row multiset — hence
/// `count(*)` and every projection — is identical.
///
/// The var-length reverse yields `(endpoint c, var-source b=slot L)` pairs; those feed
/// `reverse_walk_chain` for the fixed hops (which enforces the scan's source label on
/// slot 0). With no fixed hops (L = 0) the var-source IS the labeled source, so it is
/// filtered directly. A conjunction is seeded on its equality conjunct and residual-
/// filtered by the whole predicate. Plain hops only (no until/body_filter/both-loops,
/// no path lineage); anything else declines to the forward path.
fn try_reverse_varlen(pred: &Expr, input: &Plan, store: &Store, track: bool) -> Option<Batch> {
    if track {
        return None; // a path-reading query keeps the forward walk (lineage)
    }
    let Plan::VarLength {
        input: vl_in,
        from: vl_from,
        dir,
        edge_label,
        min,
        max,
        mode,
        until,
        body_filter,
        double_loops,
    } = input
    else {
        return None;
    };
    if until.is_some() || body_filter.is_some() || *double_loops {
        return None; // Gremlin until()/body-filter/both() have no simple reverse
    }
    // Unwrap the fixed hops below the var-length (0+ plain expands over a scan),
    // outermost-in so their `from` runs L-1 down to 0.
    let mut chain: Vec<(usize, Dir, &[String])> = Vec::new();
    let mut cur = vl_in.as_ref();
    let src_label = loop {
        match cur {
            Plan::Expand {
                input: inner,
                from,
                dir,
                edge_label,
                bind_edge: false,
                double_loops: false,
            } => {
                chain.push((*from, *dir, edge_label.as_slice()));
                cur = inner.as_ref();
            }
            Plan::Scan { label } => break label.clone(),
            _ => return None, // source must bottom at an unfiltered scan
        }
    };
    let fixed = chain.len();
    // The var-length must expand from the fixed chain's last slot, and the chain must be a
    // straight run (from L-1, L-2, …, 0). The endpoint lands in slot L+1.
    if *vl_from != fixed
        || chain
            .iter()
            .enumerate()
            .any(|(i, (from, _, _))| *from != fixed - 1 - i)
    {
        return None;
    }
    let ep_slot = fixed + 1;
    // Seed the endpoint from the index (equality / range / IN / OR / more selective AND
    // conjunct); the residual filter (below) exacts the answer.
    let bucket = seed_bucket(pred, ep_slot, store)?;
    // Cardinality decision: flip only when the endpoint bucket is smaller than the source
    // scan (the same guard as the fixed-length seed).
    let source_rows = match &src_label {
        Some(l) => store.nodes_with_label(l).len(),
        None => store.live_node_count(),
    };
    if bucket.len() >= source_rows {
        return None;
    }
    // The fixed hops are plain (no bound edge), innermost-first — the var-length output is
    // all nodes, so the reverse-walk over them produces a node-only layout.
    let mut fixed_hops: Vec<RevHop> = Vec::with_capacity(fixed);
    for &(_, dir, edge_label) in chain.iter().rev() {
        let want = match want_etypes(store, edge_label) {
            Ok(w) => w,
            Err(()) => return None,
        };
        fixed_hops.push(RevHop {
            dir,
            want,
            bind_edge: false,
        });
    }
    // Walk the var-length in reverse from the endpoint bucket. Decline (forward path) if
    // the reverse walk itself trips the trail guard — a huge-in-degree endpoint.
    let seed = Batch::of(vec![Col::Nodes(bucket)]);
    let rev = var_length(
        &seed,
        store,
        0,
        reverse_dir(*dir),
        edge_label,
        *min,
        *max,
        *mode,
        &[],
        None,
        1,
        None,
        None,
        false,
    )
    .ok()?;
    // rev is [endpoint(c) gathered, ends(b = var-source = slot L)]. Build rows headed by
    // b (the frontier for the fixed-chain reverse) with c as the suffix. With no fixed
    // hops, b IS the source, so apply the scan label here; otherwise the chain applies it.
    let (Col::Nodes(cs), Col::Nodes(bs)) = (rev.slot(0), rev.slot(1)) else {
        return None;
    };
    let rows0: Vec<Vec<u32>> = bs
        .iter()
        .zip(cs.iter())
        .filter(|(&b, _)| fixed != 0 || src_label.as_deref().is_none_or(|l| store.is_labeled(b, l)))
        .map(|(&b, &c)| vec![b, c])
        .collect();
    let rows = reverse_walk_chain(rows0, &fixed_hops, src_label.as_deref(), store);
    let out = rows_to_batch(&rows, &vec![false; ep_slot + 1]);
    // A bare equality is fully satisfied by the seed; a conjunction needs its other
    // conjuncts applied over the (small) seeded batch.
    if target_eq(pred, ep_slot).is_some() {
        return Some(out);
    }
    let keep = residual_keep(pred, store, &out)?;
    Some(out.gather(&keep))
}

/// Keep-indices for a residual predicate over an already-materialized (small) batch —
/// the fast `try_filter_keep` pass, else a boxed `eval`. `None` if `eval` faults, so a
/// caller can decline and fall back to the forward path rather than swallow the error.
fn residual_keep(pred: &Expr, store: &Store, batch: &Batch) -> Option<Vec<usize>> {
    if let Some(keep) = try_filter_keep(pred, store, batch) {
        return Some(keep);
    }
    let mask = eval_mask(pred, store, batch).ok()?;
    Some((0..mask.len()).filter(|&i| mask[i] == Some(true)).collect())
}

/// The candidate endpoint node ids for a seedable predicate on `slot` — a DEDUPED set
/// that is a SUPERSET of the predicate's exact matches. The caller's residual filter
/// (whenever `pred` isn't a bare equality) narrows it to the exact set, so this only has
/// to over-approximate, which keeps NULL / cross-type / ordering edge cases the residual's
/// job. Handles an indexed equality (hash), a range op (range index), a positive
/// `IN [lits]` (union of hash buckets), an `OR` of seedables (union), and an `AND` (the
/// more selective conjunct's bucket). `None` when nothing on `slot` is seedable.
fn seed_bucket(pred: &Expr, slot: usize, store: &Store) -> Option<Vec<u32>> {
    if let Some(b) = seed_pure(pred, slot, store) {
        return Some(b);
    }
    if let Expr::And(l, r) = pred {
        // Two range bounds on the SAME key (one lower, one upper) seed their exact
        // intersection in one BTree walk — `k >= a AND k < b` narrows to [a, b) and a
        // contradictory pair (a > b) yields the empty set. Without this, each conjunct
        // seeds independently and we keep the wider single-bound bucket (or, when the
        // pair is unsatisfiable, materialize a large bucket only to filter it to zero).
        if let Some(ids) = seed_interval(l, r, slot, store) {
            return Some(ids);
        }
        // Seed the more selective conjunct; the residual applies the whole conjunction.
        return match (seed_bucket(l, slot, store), seed_bucket(r, slot, store)) {
            (Some(a), Some(b)) => Some(if a.len() <= b.len() { a } else { b }),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
    }
    None
}

/// Two range bounds on the SAME key with OPPOSITE directions (one lower, one upper)
/// → seed their exact intersection via a two-sided range seek. Byte-identical: the
/// reverse-seed re-applies the full conjunction as its residual, and the intersection
/// is exactly the set satisfying both bounds (empty when contradictory). Same-direction
/// pairs fall through to the generic per-conjunct seed, which already picks the tighter.
fn seed_interval(l: &Expr, r: &Expr, slot: usize, store: &Store) -> Option<Vec<u32>> {
    use crate::ir::CompareOp::{Ge, Gt, Le, Lt};
    use std::ops::Bound::{Excluded, Included, Unbounded};
    let (kl, ol, vl) = endpoint_range(l, slot)?;
    let (kr, or, vr) = endpoint_range(r, slot)?;
    if kl != kr || !store.has_range_index(&kl) {
        return None;
    }
    let side = |op: CompareOp, v: &Value| match op {
        Gt => Some((true, Excluded(v.clone()))),
        Ge => Some((true, Included(v.clone()))),
        Lt => Some((false, Excluded(v.clone()))),
        Le => Some((false, Included(v.clone()))),
        _ => None,
    };
    let (l_low, lb) = side(ol, &vl)?;
    let (r_low, rb) = side(or, &vr)?;
    let (lo, hi) = match (l_low, r_low) {
        (true, false) => (lb, rb),
        (false, true) => (rb, lb),
        _ => return None, // same direction — generic seed picks the tighter conjunct
    };
    let lo_ref = match &lo {
        Included(v) => Included(v),
        Excluded(v) => Excluded(v),
        Unbounded => Unbounded,
    };
    let hi_ref = match &hi {
        Included(v) => Included(v),
        Excluded(v) => Excluded(v),
        Unbounded => Unbounded,
    };
    store.range_between(&kl, lo_ref, hi_ref)
}

/// Is a reverse-seed over a FIXED-hop chain worth it? The forward count/agg folds during
/// a single scan (cheap, no materialization); the reverse-seed materializes the walked
/// rows and boxes any residual over them. When the forward predicate ALSO folds cheaply
/// (a simple range/IN/AND that `try_filter_keep` handles), the reverse only wins on a
/// SMALL FRACTION of the scan — require its fan-out (bucket × degree²) to stay under the
/// forward scan. Only a `loose` predicate (a bare equality — no residual) wins on any
/// bucket smaller than the scan; range/IN/OR/AND all take the tight guard, so a large OR
/// union does not flip when a downstream LIMIT could stream the forward walk cheaply.
/// (Var-length seeds skip this — their forward path is a trail-limit blow-up.)
fn reverse_seed_worth(bucket: usize, source: usize, loose: bool, store: &Store) -> bool {
    if bucket >= source {
        return false;
    }
    if loose {
        return true;
    }
    // The reverse-seed materializes ~bucket × reverse-degree rows; it wins when that stays
    // under the forward scan of `source`. `deg` is the GLOBAL avg degree — deliberately: a
    // per-edge-type degree was TRIED (2026-08-15) and REVERTED, because the sparse edge type
    // here (F, deg 1) made a NON-selective range bucket (`score >= 89` = 91% of nodes) pass
    // `bucket × 1 < source` and wrongly fire the seed, materializing ~all the graph (26ms vs
    // the 3.6ms forward win). The global degree correctly declines those. The residual is
    // vectorized (`eval_mask`), so a single `deg` factor (not `deg²`) already admits the
    // genuinely selective ranges while declining the dense/non-selective ones.
    let deg = (store.edge_count() as f64 / store.live_node_count().max(1) as f64).max(1.0);
    (bucket as f64) * deg < source as f64
}

/// A predicate whose ENTIRE match set is captured by one index bucket or a union of them
/// (equality / range / positive `IN` / `OR` of such) — the deduped candidate set. Never
/// descends an `AND` (that needs a residual only the caller applies).
fn seed_pure(pred: &Expr, slot: usize, store: &Store) -> Option<Vec<u32>> {
    if let Some((k, v)) = target_eq(pred, slot) {
        if store.has_hash_index(&k) {
            return store.index_lookup(&k, &v);
        }
    }
    if let Some((k, op, v)) = endpoint_range(pred, slot) {
        if store.has_range_index(&k) {
            return store.range_lookup(&k, op, &v);
        }
    }
    if let Some((k, vals)) = endpoint_in(pred, slot) {
        if store.has_hash_index(&k) {
            let mut ids = Vec::new();
            for v in &vals {
                ids.extend(store.index_lookup(&k, v)?);
            }
            ids.sort_unstable();
            ids.dedup();
            return Some(ids);
        }
    }
    if let Expr::Or(l, r) = pred {
        let mut a = seed_pure(l, slot, store)?;
        a.extend(seed_pure(r, slot, store)?);
        a.sort_unstable();
        a.dedup();
        return Some(a);
    }
    None
}

/// `slot.key <op> lit` (or the mirror) with a range op — the endpoint-range analogue of
/// [`target_eq`]. The mirror flips the operator so `lit <op> slot.key` reads as the seek.
fn endpoint_range(pred: &Expr, slot: usize) -> Option<(String, CompareOp, Value)> {
    let Expr::Compare { op, left, right } = pred else {
        return None;
    };
    if !matches!(
        op,
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
    ) {
        return None;
    }
    let flip = |o: CompareOp| match o {
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Ge => CompareOp::Le,
        other => other,
    };
    match (left.as_ref(), right.as_ref()) {
        (Expr::Prop { slot: s, key }, Expr::Lit(v)) if *s == slot => {
            (!v.is_null()).then(|| (key.clone(), *op, v.clone()))
        }
        (Expr::Lit(v), Expr::Prop { slot: s, key }) if *s == slot => {
            (!v.is_null()).then(|| (key.clone(), flip(*op), v.clone()))
        }
        _ => None,
    }
}

/// `slot.key IN [lit, lit, …]` → (key, literal values). Positive `IN` over a literal list
/// only (a `NOT … IN` is not a superset seed); every element must be a non-null literal.
fn endpoint_in(pred: &Expr, slot: usize) -> Option<(String, Vec<Value>)> {
    let Expr::In { needle, haystack } = pred else {
        return None;
    };
    let Expr::Prop { slot: s, key } = needle.as_ref() else {
        return None;
    };
    if *s != slot {
        return None;
    }
    let Expr::List { items } = haystack.as_ref() else {
        return None;
    };
    let mut vals = Vec::with_capacity(items.len());
    for it in items {
        let Expr::Lit(v) = it else {
            return None;
        };
        if v.is_null() {
            return None;
        }
        vals.push(v.clone());
    }
    (!vals.is_empty()).then(|| (key.clone(), vals))
}

/// Parse `Prop{slot, key} = Lit(value)` (or its mirror) — an equality on the given slot.
fn target_eq(pred: &Expr, slot: usize) -> Option<(String, Value)> {
    let Expr::Compare {
        op: CompareOp::Eq,
        left,
        right,
    } = pred
    else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (Expr::Prop { slot: s, key }, Expr::Lit(v))
        | (Expr::Lit(v), Expr::Prop { slot: s, key })
            if *s == slot =>
        {
            (!v.is_null()).then(|| (key.clone(), v.clone()))
        }
        _ => None,
    }
}

/// `keep`, so no per-row struct is built.
fn expand(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: &[String],
    bind_edge: bool,
    double_loops: bool,
) -> Batch {
    // An empty expand still appends the landed slot(s), so the output has the same
    // shape a successful expand would (K+1 slots, or K+2 with the edge bound) — a
    // projection referencing a new slot must not go out of bounds.
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        if bind_edge {
            slots.push(Col::Edges(vec![]));
        }
        slots.push(Col::Nodes(vec![]));
        let mut b = Batch::of(slots);
        if batch.lineage.is_some() {
            b.lineage = Some(Lineage::empty());
        }
        b
    };
    // Resolve the edge label to an interned id up front; an unknown label matches
    // nothing (not everything).
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return empty(),
    };
    let Col::Nodes(src) = batch.slot(from) else {
        // Only a node frontier can be expanded; anything else yields nothing.
        return empty();
    };

    // Collect edge ids only when something needs them — a bound edge slot or
    // lineage — so the lineage-free hot path pushes nothing extra per neighbour.
    let track = batch.lineage.is_some();
    let need_eids = bind_edge || track;
    let mut keep = Vec::new();
    let mut nbrs = Vec::new();
    let mut eids = Vec::new();
    for (row, &v) in src.iter().enumerate() {
        for_each_nbr(store, v, dir, &want, double_loops, |nbr, eid| {
            keep.push(row);
            nbrs.push(nbr);
            if need_eids {
                eids.push(eid);
            }
        });
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    if bind_edge {
        slots.push(Col::Edges(eids.clone())); // edge slot at index W
    }
    slots.push(Col::Nodes(nbrs.clone())); // node slot at index W (or W+1)
    let mut out = Batch::of(slots);
    // Lineage strategy: when the input carried a path, extend each output row's
    // path by the neighbour it landed on AND the edge it crossed, so both
    // `nodes(p)` and `relationships(p)` are recoverable.
    if let Some(lin) = &batch.lineage {
        out.lineage = Some(lin.extend(&keep, &nbrs, &eids));
    }
    out
}

/// LEFT-OUTER single hop (`Plan::OptionalExpand`, GQL `OPTIONAL MATCH`): like
/// [`expand`], but a source row with NO matching neighbour is KEPT, its appended
/// node slot holding the `u32::MAX` null sentinel (read back as NULL everywhere).
/// Node-only, no lineage. So every input row yields at least one output row.
fn optional_expand(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: &[String],
    keep_source: bool,
    bind_edge: bool,
) -> Batch {
    // The value a missed row lands: the source element (Gremlin optional) or the
    // null sentinel (GQL OPTIONAL MATCH). `miss(v)` picks per row.
    let miss = |v: u32| if keep_source { v } else { u32::MAX };
    // Every left row gets exactly one neighbour-less row — used when the edge type
    // is unknown, or the `from` slot isn't a node frontier.
    let all_miss = || {
        let keep: Vec<usize> = (0..batch.rows()).collect();
        let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
        let landed: Vec<u32> = match (keep_source, batch.slot(from)) {
            (true, Col::Nodes(src)) => src.clone(),
            _ => vec![u32::MAX; batch.rows()],
        };
        if bind_edge {
            slots.push(Col::Edges(vec![u32::MAX; batch.rows()])); // no edge on a miss
        }
        slots.push(Col::Nodes(landed));
        Batch::of(slots)
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return all_miss(), // unknown edge type → no match for any row
    };
    let Col::Nodes(src) = batch.slot(from) else {
        return all_miss();
    };
    let mut keep = Vec::new();
    let mut nbrs = Vec::new();
    let mut eids = Vec::new();
    for (row, &v) in src.iter().enumerate() {
        let before = nbrs.len();
        for_each_nbr(store, v, dir, &want, false, |nbr, eid| {
            keep.push(row);
            nbrs.push(nbr);
            eids.push(eid);
        });
        if nbrs.len() == before {
            // No neighbour — keep the row, landing the miss value (source or null),
            // and a null-sentinel edge if the edge is bound.
            keep.push(row);
            nbrs.push(miss(v));
            eids.push(u32::MAX);
        }
    }
    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    if bind_edge {
        slots.push(Col::Edges(eids)); // edge column BEFORE the node column
    }
    slots.push(Col::Nodes(nbrs));
    Batch::of(slots)
}

/// Interval-overlap hop (`Plan::IntervalExpand`): like [`expand`], but keeps only
/// edges whose `[lo_key, hi_key]` interval overlaps `[qlo, qhi]` (per input row).
/// Seek-or-scan: an OUT hop over a store with a matching interval index seeks
/// (`for_each_overlap`); otherwise it scans the adjacency and applies the overlap
/// itself — the rows are identical either way, so the optimizer can fuse without
/// knowing whether the index exists. A non-numeric/absent bound or edge interval
/// yields no match for that edge (matching what the `<=`/`>=` filter would do).
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors `expand` plus the two bounds and keys"
)]
fn interval_expand(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: &[String],
    lo_key: &str,
    hi_key: &str,
    qlo_col: &Col,
    qhi_col: &Col,
    bind_edge: bool,
) -> Batch {
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        if bind_edge {
            slots.push(Col::Edges(vec![]));
        }
        slots.push(Col::Nodes(vec![]));
        let mut b = Batch::of(slots);
        if batch.lineage.is_some() {
            b.lineage = Some(Lineage::empty());
        }
        b
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return empty(),
    };
    let Col::Nodes(src) = batch.slot(from) else {
        return empty();
    };
    // Seek only an OUT hop over a matching index (the index is over out-edges);
    // any other case scans and applies the overlap.
    let can_seek = matches!(dir, Dir::Out) && store.has_interval_index(lo_key, hi_key);

    let mut keep = Vec::new();
    let mut nbrs = Vec::new();
    let mut eids = Vec::new();
    for (row, &v) in src.iter().enumerate() {
        let qlo = qlo_col.value_at(row);
        let qhi = qhi_col.value_at(row);
        // A NULL bound can never satisfy the comparison, so the row contributes
        // nothing (matching what the `<=`/`>=` filter would do).
        if qlo.is_null() || qhi.is_null() {
            continue;
        }
        // The RI-tree seek is numeric-only; NUMERIC bounds over a matching OUT index
        // take it. Temporal (or index-less) bounds fall to the scan below.
        if can_seek {
            if let (Value::Num(qlo_n), Value::Num(qhi_n)) = (&qlo, &qhi) {
                store.for_each_overlap(v, *qlo_n, *qhi_n, |eid, nbr| {
                    keep.push(row);
                    nbrs.push(nbr);
                    eids.push(eid);
                });
                continue;
            }
        }
        // General scan: keep an edge whose `[lo, hi]` overlaps `[qlo, qhi]` — `lo <=
        // qhi AND hi >= qlo` under the value contract's 3VL ordering (`cmp_partial`),
        // the SAME comparison the `<=`/`>=` filter uses. So numeric AND temporal
        // bounds work, and an absent / incomparable bound drops out exactly as the
        // filter would (an unknown comparison is not an overlap).
        for_each_nbr(store, v, dir, &want, false, |nbr, eid| {
            let lo = store.edge_prop(eid, lo_key);
            let hi = store.edge_prop(eid, hi_key);
            let lo_le_qhi = value::cmp_partial(&lo, &qhi).map(std::cmp::Ordering::is_le);
            let hi_ge_qlo = value::cmp_partial(&hi, &qlo).map(std::cmp::Ordering::is_ge);
            if lo_le_qhi == Some(true) && hi_ge_qlo == Some(true) {
                keep.push(row);
                nbrs.push(nbr);
                eids.push(eid);
            }
        });
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    if bind_edge {
        slots.push(Col::Edges(eids.clone()));
    }
    slots.push(Col::Nodes(nbrs.clone()));
    let mut out = Batch::of(slots);
    if let Some(lin) = &batch.lineage {
        out.lineage = Some(lin.extend(&keep, &nbrs, &eids));
    }
    out
}

/// The nodes with `label` whose property `key` equals `value` under predicate
/// `=` — the rows an `IndexSeek` produces. Uses a property index when present
/// (candidates intersected with the label), else scans the label and filters by
/// `value::equals`. A NaN/NULL literal matches nothing (as `=` does).
fn index_seek_ids(store: &Store, label: &str, key: &str, value: &Value) -> Vec<u32> {
    if value.is_null() || matches!(value, Value::Num(x) if x.is_nan()) {
        return Vec::new();
    }
    match store.index_lookup(key, value) {
        Some(cands) => {
            // group_key == equals for a finite, non-null value, so the index bucket
            // is exact; intersect with the label. The label bucket is sorted
            // ascending, so binary-search each (usually few) candidate rather than
            // building a HashSet of the WHOLE label per query — that O(label) build
            // made an indexed seek SLOWER than the typed scan it was meant to beat.
            let bucket = store.nodes_with_label(label);
            cands
                .into_iter()
                .filter(|id| bucket.binary_search(id).is_ok())
                .collect()
        }
        None => {
            let ids = store.nodes_with_label(label);
            // Typed fast paths for a plain (non-dotted) key: compare the raw column
            // — a `&str`/`f64`/`bool` compare, no per-cell `Value` boxing or `Arc`
            // clone. Equality semantics match `value::equals` (a present cell of the
            // literal's type; a NULL cell — `present == false` — never equals).
            if !key.contains('.') {
                match (store.column(key), value) {
                    (Some(Column::Str { data, present, .. }), Value::Str(t)) => {
                        let t: &str = t;
                        return ids
                            .iter()
                            .copied()
                            .filter(|&id| present[id as usize] && &*data[id as usize] == t)
                            .collect();
                    }
                    (
                        Some(Column::Dict {
                            dict,
                            codes,
                            present,
                            ..
                        }),
                        Value::Str(t),
                    ) => {
                        // Resolve the literal to its code ONCE, then match rows on the
                        // `u32` — no per-row string compare. A literal absent from the
                        // dict matches nothing.
                        let t: &str = t;
                        let Some(want) = dict.iter().position(|s| &**s == t) else {
                            return Vec::new();
                        };
                        let want = want as u32;
                        return ids
                            .iter()
                            .copied()
                            .filter(|&id| present[id as usize] && codes[id as usize] == want)
                            .collect();
                    }
                    (Some(Column::Num { data, present, .. }), Value::Num(t)) => {
                        let t = *t;
                        return ids
                            .iter()
                            .copied()
                            .filter(|&id| present[id as usize] && data[id as usize] == t)
                            .collect();
                    }
                    (Some(Column::Bool { data, present, .. }), Value::Bool(t)) => {
                        let t = *t;
                        return ids
                            .iter()
                            .copied()
                            .filter(|&id| present[id as usize] && data[id as usize] == t)
                            .collect();
                    }
                    _ => {}
                }
            }
            // `key` may be a dotted record path — resolve it (plain keys read as
            // `prop`), so the no-index fallback matches a dotted seek too.
            ids.iter()
                .copied()
                .filter(|&id| value::equals(&store.prop_path(id, key), value))
                .collect()
        }
    }
}

/// Whether `prop <op> value` holds — the exact test the `Filter` executor applies
/// for a range comparison. Three-valued via `cmp_partial`: a NULL operand OR
/// incomparable operands (different types / NaN) are UNKNOWN → false (the row
/// drops), matching the general `compare` path. `op` must be a range op; `value`
/// is non-null.
fn range_pass(prop: &Value, op: CompareOp, value: &Value) -> bool {
    if prop.is_null() {
        return false;
    }
    let Some(ord) = value::cmp_partial(prop, value) else {
        return false; // incomparable → UNKNOWN → drop
    };
    match op {
        CompareOp::Lt => ord.is_lt(),
        CompareOp::Le => ord.is_le(),
        CompareOp::Gt => ord.is_gt(),
        CompareOp::Ge => ord.is_ge(),
        CompareOp::Eq | CompareOp::Ne => false,
    }
}

/// The nodes with `label` whose property `key` satisfies `key <op> value` — the
/// rows a `RangeSeek` produces. Uses a range index when present (candidates
/// intersected with the label), else scans and filters via `range_pass`. A NULL
/// `value` matches nothing (predicate UNKNOWN), matching a scan+filter.
/// Raw-f64 comparison matching the value contract for two present numbers: `==`/
/// `!=` for equality, `<`/`<=`/`>`/`>=` for ordering (a NaN operand makes ordering
/// false — 3VL "unknown → drop", as `cmp_partial` gives). Used by the typed
/// scan/filter fast paths so a numeric predicate never boxes a `Value`.
fn num_pred(op: CompareOp, x: f64, t: f64) -> bool {
    match op {
        CompareOp::Eq => x == t,
        CompareOp::Ne => x != t,
        CompareOp::Lt => x < t,
        CompareOp::Le => x <= t,
        CompareOp::Gt => x > t,
        CompareOp::Ge => x >= t,
    }
}

/// Byte-lexicographic string comparison — the same order `value::cmp_partial`/`equals`
/// give two present `Str`/`Dict` values, so a typed leaf matches the boxed `compare`.
fn str_pred(op: CompareOp, a: &str, b: &str) -> bool {
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Ne => a != b,
        CompareOp::Lt => a < b,
        CompareOp::Le => a <= b,
        CompareOp::Gt => a > b,
        CompareOp::Ge => a >= b,
    }
}

fn range_seek_ids(store: &Store, label: &str, key: &str, op: CompareOp, value: &Value) -> Vec<u32> {
    if value.is_null() {
        return Vec::new();
    }
    match store.range_lookup(key, op, value) {
        Some(cands) => {
            // Test label membership with a per-candidate binary search of the sorted
            // label bucket (O(cands·log|label|)), NOT a HashSet built from the whole
            // bucket — that build is O(|label|) and dominates when the label covers
            // most of the graph (a single-label store makes it pure waste: the range
            // index already narrowed to `cands`, then we'd rebuild a set of everything
            // to intersect back down). When the bucket covers ALL non-deleted nodes,
            // every candidate is in-label, so skip the test entirely.
            let bucket = store.nodes_with_label(label);
            let all_in_label = bucket.len() == store.live_node_count();
            cands
                .into_iter()
                .filter(|&id| all_in_label || store.is_labeled(id, label))
                // The index orders by the TOTAL order (cross-type by rank), but the
                // OPERATOR is three-valued (cross-type → UNKNOWN → drop). Re-check
                // each candidate with `range_pass` so an indexed seek returns
                // exactly the scan-filter rows (the equivalent-spellings invariant);
                // for a homogeneous column this keeps every candidate.
                .filter(|&id| range_pass(&store.prop(id, key), op, value))
                .collect()
        }
        None => {
            let ids = store.nodes_with_label(label);
            // Typed fast path: a Num column vs a Num bound compares RAW f64 (no
            // per-cell Value boxing) — the no-index scan is the common case.
            if let (Some(Column::Num { data, present, .. }), Value::Num(t)) =
                (store.column(key), value)
            {
                let t = *t;
                return ids
                    .iter()
                    .copied()
                    .filter(|&id| present[id as usize] && num_pred(op, data[id as usize], t))
                    .collect();
            }
            ids.iter()
                .copied()
                .filter(|&id| range_pass(&store.prop(id, key), op, value))
                .collect()
        }
    }
}

/// Visit each neighbour of `v` along `dir` matching edge type `want`, calling `f`
/// with `(neighbour, eid)`. The one place Expand's adjacency walk is spelled —
/// shared by the batch operator and the frontier executor so the two can never
/// disagree on what an Expand reaches.
/// Resolve an edge-type constraint (the plan's `edge_label` list) to the matching
/// type ids the walkers filter on. An EMPTY list is untyped — any edge — so it
/// returns `Ok(vec![])` (an empty `want` slice reads as "any"). A typed list whose
/// names ALL fail to resolve matches no edge, so it returns `Err(())` and the caller
/// short-circuits to its own empty result. Otherwise the known ids, unknown names
/// dropped — mirroring core's `lower_labels`.
/// Does node `v` have ANY neighbour over `dir`/`want` (empty `want` = any type)? A
/// short-circuiting existence check for `where(out/in/both)` — scans adjacency until
/// the first match, never materializing the neighbours. Self-loop doubling is moot for
/// existence.
/// A simple predicate on the neighbour a `where(out()/in()/both())` hop lands on — the
/// leaf inside `Exists { Filter{pred} over Expand }`. Extracted so the semijoin can test
/// it per neighbour with early-stop instead of expanding the whole frontier.
enum NbrPred {
    Cmp(String, CompareOp, Value),
    Exists(String),
    Labeled(Vec<String>),
}

/// Recognize a single leaf predicate on the hop's neighbour (a property compare, a
/// presence test, or a label test), referencing a slot OTHER than the source `from` — so
/// it is the appended endpoint. `None` for anything compound/unsupported (→ general path).
fn simple_nbr_pred(pred: &Expr, from: usize) -> Option<NbrPred> {
    match pred {
        Expr::Compare { op, left, right } => {
            let (slot, key, v, flip) = match (left.as_ref(), right.as_ref()) {
                (Expr::Prop { slot, key }, Expr::Lit(v)) => (*slot, key.clone(), v.clone(), false),
                (Expr::Lit(v), Expr::Prop { slot, key }) => (*slot, key.clone(), v.clone(), true),
                _ => return None,
            };
            if slot == from || v.is_null() {
                return None;
            }
            let op = if flip {
                match op {
                    CompareOp::Lt => CompareOp::Gt,
                    CompareOp::Le => CompareOp::Ge,
                    CompareOp::Gt => CompareOp::Lt,
                    CompareOp::Ge => CompareOp::Le,
                    other => *other,
                }
            } else {
                *op
            };
            Some(NbrPred::Cmp(key, op, v))
        }
        Expr::PropertyExists { slot, key } if *slot != from => Some(NbrPred::Exists(key.clone())),
        Expr::IsLabeled { slot, labels } if *slot != from => Some(NbrPred::Labeled(labels.clone())),
        // `<label> IN labels(m)` — the `(m:Label)` test, UNNORMALIZED inside an EXISTS body
        // (the optimizer doesn't descend into subquery bodies), treated as a label test.
        Expr::In { needle, haystack } => {
            if let (Expr::Lit(Value::Str(l)), Expr::Call { name, args }) =
                (needle.as_ref(), haystack.as_ref())
            {
                if name == "labels" && args.len() == 1 {
                    if let Expr::Slot(s) = args[0] {
                        if s != from {
                            return Some(NbrPred::Labeled(vec![l.to_string()]));
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// A CONJUNCTION of simple neighbour predicates — `has(k,v)` plus the `(m:Label)` label
/// test the GQL `EXISTS { (n)->(m:Label) WHERE … }` and `(b:Label)` patterns add. Every
/// leaf of the `And`-tree must be a simple neighbour predicate, else `None`.
fn simple_nbr_preds(pred: &Expr, from: usize) -> Option<Vec<NbrPred>> {
    fn collect(e: &Expr, from: usize, out: &mut Vec<NbrPred>) -> bool {
        match e {
            Expr::And(a, b) => collect(a, from, out) && collect(b, from, out),
            other => match simple_nbr_pred(other, from) {
                Some(p) => {
                    out.push(p);
                    true
                }
                None => false,
            },
        }
    }
    let mut out = Vec::new();
    collect(pred, from, &mut out).then_some(())?;
    (!out.is_empty()).then_some(out)
}

/// Does node `nbr` satisfy the leaf (3VL: only a definite TRUE counts, matching the body
/// filter's "keep if TRUE" — a NULL operand is UNKNOWN, not a match).
fn nbr_pred_ok(store: &Store, nbr: u32, p: &NbrPred) -> bool {
    match p {
        NbrPred::Exists(k) => store.has_prop(nbr, k),
        NbrPred::Labeled(ls) => ls.iter().any(|l| store.is_labeled(nbr, l)),
        NbrPred::Cmp(k, op, v) => {
            let a = store.prop(nbr, k);
            if a.is_null() {
                return false;
            }
            match op {
                CompareOp::Eq => value::equals(&a, v),
                CompareOp::Ne => !value::equals(&a, v),
                other => value::cmp_partial(&a, v).is_some_and(|o| match other {
                    CompareOp::Lt => o.is_lt(),
                    CompareOp::Le => o.is_le(),
                    CompareOp::Gt => o.is_gt(),
                    CompareOp::Ge => o.is_ge(),
                    _ => false,
                }),
            }
        }
    }
}

/// Does `v` have ANY neighbour whose (pre-resolved) numeric column satisfies `op t`?
/// Early-stops. The raw-column form of [`node_has_matching_nbr`] — one column resolution
/// for the whole query instead of a `store.prop` hash lookup per neighbour.
fn node_has_num_nbr(
    store: &Store,
    v: u32,
    dir: Dir,
    want: &[u32],
    col: (&[f64], &[bool]),
    cmp: (CompareOp, f64),
) -> bool {
    let (data, present) = col;
    let (op, t) = cmp;
    let has_extra = store.has_multi_label_edges();
    let type_ok = |et: u32, eid: u32| {
        want.is_empty()
            || want
                .iter()
                .any(|&w| w == et || (has_extra && store.edge_type_matches(et, eid, w)))
    };
    let hit = |a: &crate::store::Adj| {
        type_ok(a.etype, a.eid) && present[a.nbr as usize] && num_pred(op, data[a.nbr as usize], t)
    };
    (matches!(dir, Dir::Out | Dir::Both) && store.out(v).iter().any(hit))
        || (matches!(dir, Dir::In | Dir::Both) && store.inc(v).iter().any(hit))
}

/// Does `v` have ANY neighbour (in `dir`, edge-type in `want`) satisfying ALL of `preds`?
/// Early-stops on the first match — the semijoin win over expanding every neighbour.
fn node_has_matching_nbr(store: &Store, v: u32, dir: Dir, want: &[u32], preds: &[NbrPred]) -> bool {
    let has_extra = store.has_multi_label_edges();
    let type_ok = |et: u32, eid: u32| {
        want.is_empty()
            || want
                .iter()
                .any(|&w| w == et || (has_extra && store.edge_type_matches(et, eid, w)))
    };
    let hit = |a: &crate::store::Adj| {
        type_ok(a.etype, a.eid) && preds.iter().all(|p| nbr_pred_ok(store, a.nbr, p))
    };
    (matches!(dir, Dir::Out | Dir::Both) && store.out(v).iter().any(hit))
        || (matches!(dir, Dir::In | Dir::Both) && store.inc(v).iter().any(hit))
}

fn node_has_nbr(store: &Store, v: u32, dir: Dir, want: &[u32]) -> bool {
    let has_extra = store.has_multi_label_edges();
    let type_ok = |et: u32, eid: u32| {
        // Pass the primary type `et` (already in a register from the Adj) to
        // edge_type_matches instead of edge_has_label, so a primary-type MISS does not
        // pay a redundant random read of edge_etype[eid] to re-learn `et != w` — only
        // the (rare) secondary-set probe. Guard the probe on has_extra so a single-label
        // graph never touches it. See edge_type_matches / core's 4.3x note.
        want.is_empty()
            || want
                .iter()
                .any(|&w| w == et || (has_extra && store.edge_type_matches(et, eid, w)))
    };
    if matches!(dir, Dir::Out | Dir::Both) && store.out(v).iter().any(|a| type_ok(a.etype, a.eid)) {
        return true;
    }
    if matches!(dir, Dir::In | Dir::Both) && store.inc(v).iter().any(|a| type_ok(a.etype, a.eid)) {
        return true;
    }
    false
}

fn want_etypes(store: &Store, edge_label: &[String]) -> Result<Vec<u32>, ()> {
    if edge_label.is_empty() {
        return Ok(Vec::new());
    }
    // A leading "!" sentinel marks a NEGATED label set (`-[:!T]->` / `-[:!(A|B)]->`):
    // the hop matches any edge whose type is NOT one of the named ones. Resolve to the
    // COMPLEMENT id set so every downstream membership check is unchanged. An unknown
    // named type contributes nothing to exclude; if the complement is empty (every
    // type excluded) the hop matches nothing (`Err`).
    if edge_label[0] == "!" {
        let excluded: Vec<u32> = edge_label[1..]
            .iter()
            .filter_map(|n| store.etype_id(n))
            .collect();
        let complement: Vec<u32> = store
            .all_etype_ids()
            .into_iter()
            .filter(|id| !excluded.contains(id))
            .collect();
        if complement.is_empty() {
            return Err(());
        }
        return Ok(complement);
    }
    let ids: Vec<u32> = edge_label
        .iter()
        .filter_map(|n| store.etype_id(n))
        .collect();
    if ids.is_empty() {
        return Err(());
    }
    Ok(ids)
}

/// Does adjacency entry `a` carry one of the `want` labels? Empty `want` = any
/// edge. Checks the primary type (`a.etype`, already in a register) then, only on a
/// multi-label graph, the eid's secondary set. The one predicate every edge-type
/// filter shares, so a `:Y` hop over a multi-label edge matches everywhere.
#[inline]
fn edge_carries_wanted(store: &Store, a: &crate::store::Adj, want: &[u32]) -> bool {
    want.is_empty()
        || want.iter().any(|&w| {
            w == a.etype || (store.has_multi_label_edges() && store.edge_has_label(a.eid, w))
        })
}

fn for_each_nbr(
    store: &Store,
    v: u32,
    dir: Dir,
    want: &[u32],
    double_loops: bool,
    mut f: impl FnMut(u32, u32),
) {
    // An undirected walk reaches a self-loop from BOTH the out- and the in-index.
    // GQL emits it ONCE (drops the in-side copy); Gremlin `both()` keeps BOTH
    // (`double_loops`) — the self-loop is an out-edge AND an in-edge. Directed walks
    // touch one index, so they keep it either way.
    let drop_loop = matches!(dir, Dir::Both) && !double_loops;
    // A SINGLE-type hop over an indexed store seeks the type bucket directly
    // (O(matching), not O(degree)) — the whole point of the opt-in edge-type index.
    // A disjunction (`want.len() >= 2`) must NOT union buckets: that reorders vs the
    // flat stored-order scan and would break byte-identity, so it falls through.
    // The type-index bucket keys on an edge's PRIMARY label only, so it cannot see
    // a `:Y` match on a multi-label edge whose first label is `X`. Skip it whenever
    // the graph has any multi-label edge (rare) and fall to the flat scan below,
    // which consults the secondary labels.
    if let [w] = want {
        if store.has_edge_type_index() && !store.has_multi_label_edges() {
            if matches!(dir, Dir::Out | Dir::Both) {
                for a in store.out_typed(v, *w) {
                    f(a.nbr, a.eid);
                }
            }
            if matches!(dir, Dir::In | Dir::Both) {
                for a in store.in_typed(v, *w) {
                    if !(drop_loop && a.nbr == v) {
                        f(a.nbr, a.eid);
                    }
                }
            }
            return;
        }
        // Per-type CSR fast path: a single-type hop over a single-label graph iterates ONLY
        // this node's type-`w` edges (a contiguous slice in out_adj order — byte-identical to
        // the flat scan filtering `etype == w`), so a sparse type does not pay the dense
        // types' degree. Available once the CSR overlay is fresh; a stale overlay returns None
        // and we fall through to the flat scan below.
        if !store.has_multi_label_edges() {
            match dir {
                Dir::Out => {
                    if let Some(sl) = store.out_typed_csr(v, *w) {
                        for a in sl {
                            f(a.nbr, a.eid);
                        }
                        return;
                    }
                }
                Dir::In => {
                    if let Some(sl) = store.in_typed_csr(v, *w) {
                        for a in sl {
                            if !(drop_loop && a.nbr == v) {
                                f(a.nbr, a.eid);
                            }
                        }
                        return;
                    }
                }
                Dir::Both => {
                    if let (Some(o), Some(i)) =
                        (store.out_typed_csr(v, *w), store.in_typed_csr(v, *w))
                    {
                        for a in o {
                            f(a.nbr, a.eid);
                        }
                        for a in i {
                            if !(drop_loop && a.nbr == v) {
                                f(a.nbr, a.eid);
                            }
                        }
                        return;
                    }
                }
            }
        }
    }
    // Empty `want` = any type; otherwise the edge must carry one of the wanted
    // labels. `edge_has_label` checks the primary type (already in `a.etype`) then,
    // only when the graph has multi-label edges, the eid's secondary set.
    let has_extra = store.has_multi_label_edges();
    let type_ok = |et: u32, eid: u32| {
        // Pass the primary type `et` (already in a register from the Adj) to
        // edge_type_matches instead of edge_has_label, so a primary-type MISS does not
        // pay a redundant random read of edge_etype[eid] to re-learn `et != w` — only
        // the (rare) secondary-set probe. Guard the probe on has_extra so a single-label
        // graph never touches it. See edge_type_matches / core's 4.3x note.
        want.is_empty()
            || want
                .iter()
                .any(|&w| w == et || (has_extra && store.edge_type_matches(et, eid, w)))
    };
    if matches!(dir, Dir::Out | Dir::Both) {
        for a in store.out(v) {
            if type_ok(a.etype, a.eid) {
                f(a.nbr, a.eid);
            }
        }
    }
    if matches!(dir, Dir::In | Dir::Both) {
        for a in store.inc(v) {
            if type_ok(a.etype, a.eid) && !(drop_loop && a.nbr == v) {
                f(a.nbr, a.eid);
            }
        }
    }
}

/// Whether a hop's wanted-type set `want` matches EVERY edge (so a count needs no
/// per-edge type check): an empty `want` (any type), or a set that covers all
/// interned edge types. `want` is always a distinct subset of the etype ids, so a
/// length match is the fast test; the `all` probe is a cheap guard against a
/// degenerate duplicate (`-[:R|R]->`) inflating the length. Computed ONCE per
/// count, not per node.
fn want_covers_all_etypes(store: &Store, want: &[u32]) -> bool {
    want.is_empty()
        || (want.len() == store.num_etypes()
            && store.all_etype_ids().iter().all(|id| want.contains(id)))
}

/// `v`'s matching out/in degree as an f64 — the number of times `for_each_nbr`
/// would fire for this node — WITHOUT walking each edge when `all_types` says the
/// type filter is trivially satisfied: a directed hop is then the raw adjacency
/// length (one read, no per-edge type check). `Dir::Both` keeps the walk (it dedups
/// the in-side self-loop copy), as does any partial want. Byte-identical in VALUE.
fn matching_degree(
    store: &Store,
    v: u32,
    dir: Dir,
    want: &[u32],
    double_loops: bool,
    all_types: bool,
) -> f64 {
    if all_types {
        match dir {
            Dir::Out => return store.out(v).len() as f64,
            Dir::In => return store.inc(v).len() as f64,
            Dir::Both => {}
        }
    }
    let mut deg = 0f64;
    for_each_nbr(store, v, dir, want, double_loops, |_, _| deg += 1.0);
    deg
}

/// Slot count of a pure Scan/Expand chain; `None` for anything else (Filter,
/// Join, VarLength, …). The frontier executor only handles such chains.
fn chain_width(plan: &Plan) -> Option<usize> {
    match plan {
        // A seek, like a scan, seeds a single-slot frontier.
        Plan::Scan { .. } | Plan::IndexSeek { .. } | Plan::RangeSeek { .. } => Some(1),
        // A bind_edge Expand appends TWO slots (edge then node), else one.
        Plan::Expand {
            input, bind_edge, ..
        } => Some(chain_width(input)? + if *bind_edge { 2 } else { 1 }),
        // A frontier `hasLabel(…)` filter keeps the width (it drops rows, not slots) —
        // so a fused counter can see the chain through it (see `frontier_counts`).
        Plan::Filter {
            input,
            pred: Expr::IsLabeled { .. },
        } => chain_width(input),
        _ => None,
    }
}

/// Slot count of a Scan/Expand chain, seeing through ANY row-filter — safe ONLY for an
/// executor that PULLS the chain (the pull applies every filter, so the frontier is already
/// filtered). The count-fold paths must NOT use this (they re-walk and re-check the filter
/// themselves, so they stay on the IsLabeled-only `chain_width`); the DISTINCT frontier
/// paths, which pull, can.
fn chain_pull_width(plan: &Plan) -> Option<usize> {
    match plan {
        Plan::Scan { .. } | Plan::IndexSeek { .. } | Plan::RangeSeek { .. } => Some(1),
        Plan::Expand {
            input, bind_edge, ..
        } => Some(chain_pull_width(input)? + if *bind_edge { 2 } else { 1 }),
        // A plain var-length hop appends one endpoint slot (no bound edge / groups here).
        Plan::VarLength { input, .. } => Some(chain_pull_width(input)? + 1),
        Plan::Filter { input, .. } => chain_pull_width(input),
        _ => None,
    }
}

/// The current node frontier of a pure Scan/Expand chain — the last slot's node
/// ids, WITH multiplicity (one entry per path reaching the node) — produced
/// without ever materializing the earlier slots. `None` if the plan is not such
/// a chain. This is the batch model's payoff: when nothing above the chain reads
/// an earlier slot, the chain need only carry its frontier.
///
/// Rejected optimization: replacing this Vec with a streaming `for_each_frontier`
/// callback (so the fused counts never build the intermediate at all). It had to
/// pass the callback as `&mut dyn FnMut` — a generic bound blows monomorphization
/// on the recursion — and the resulting per-node indirect call, nested one level
/// per hop, cost MORE than building and rescanning the vector: at 1M/8 it moved
/// 2-hop count(*) 40->64ms and count(DISTINCT) 54->62ms. The sequential Vec push
/// is cheap; per-element dynamic dispatch over tens of millions of nodes is not.
fn frontier_ids(plan: &Plan, store: &Store) -> Option<Vec<u32>> {
    match plan {
        Plan::Scan { label } => Some(match label {
            Some(l) => store.nodes_with_label(l).to_vec(),
            None => store.all_nodes(),
        }),
        Plan::IndexSeek { label, key, value } => Some(index_seek_ids(store, label, key, value)),
        Plan::RangeSeek {
            label,
            key,
            op,
            value,
        } => Some(range_seek_ids(store, label, key, *op, value)),
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            double_loops,
            ..
        } => {
            // Must expand the CURRENT frontier (the last slot); a linear pattern
            // always does, but a hand-built plan might not.
            if *from + 1 != chain_width(input)? {
                return None;
            }
            let src = frontier_ids(input, store)?;
            let want = match want_etypes(store, edge_label) {
                Ok(w) => w,
                Err(()) => return Some(Vec::new()), // unknown label matches nothing
            };
            // Gremlin `both()` walks a self-loop TWICE (`double_loops`), so the endpoint
            // multiset matches core.
            let dl = *double_loops;
            let mut out = Vec::new();
            for &v in &src {
                for_each_nbr(store, v, *dir, &want, dl, |nbr, _eid| out.push(nbr));
            }
            Some(out)
        }
        _ => None,
    }
}

/// The number of `Expand` hops in a pure Scan/Expand chain (0 for a bare seed).
fn count_hops(plan: &Plan) -> usize {
    match plan {
        Plan::Sample { input, .. }
        | Plan::Enumerate { input, .. }
        | Plan::EdgeVertex { input, .. }
        | Plan::Expand { input, .. } => 1 + count_hops(input),
        _ => 0,
    }
}

/// A per-node PATH-COUNT frontier, stored SPARSE (a list of active `(node, count)`)
/// while few nodes carry a path, DENSE (indexed by node id) once the frontier
/// covers a large fraction of the graph. A 5-hop count from a SINGLE source touches
/// at most fan-out^hops distinct nodes — kept sparse, it costs O(active) per hop
/// instead of the O(node_count) alloc + full scan a dense array pays every hop
/// (that made `aml/chain5` 30x SLOWER than core: an 8 MB zeroed array and a 1M-entry
/// scan, five times, for a frontier of a few hundred nodes). Counts are exact
/// integers (< 2^53), so the f64 sums are order-independent and the representation
/// switch is byte-identical.
enum Counts {
    Sparse(Vec<(u32, f64)>),
    Dense(Vec<f64>),
}

impl Counts {
    /// Call `f(node, count)` for every node carrying a non-zero count.
    fn for_each(&self, mut f: impl FnMut(u32, f64)) {
        match self {
            Counts::Sparse(v) => {
                for &(id, c) in v {
                    f(id, c);
                }
            }
            Counts::Dense(a) => {
                for (i, &c) in a.iter().enumerate() {
                    if c != 0.0 {
                        f(i as u32, c);
                    }
                }
            }
        }
    }

    /// Number of nodes carrying a path — the cost driver for the next hop.
    fn active(&self) -> usize {
        match self {
            Counts::Sparse(v) => v.len(),
            Counts::Dense(a) => a.iter().filter(|&&c| c != 0.0).count(),
        }
    }
}

/// The per-node PATH-COUNT frontier of a pure Scan/Expand chain: `counts[v]` is the
/// number of chain paths whose last node is `v`. Propagated one hop at a time
/// (`next[nbr] += counts[v]` over each matching edge) so it never materializes the
/// exploding path multiset that [`frontier_ids`] carries — O(hops * edges) time.
/// Sparse until the frontier is large (see [`Counts`]), so a narrow deep chain pays
/// O(active) not O(node_count) per hop. `None` for a non-chain.
fn frontier_counts(plan: &Plan, store: &Store) -> Option<Counts> {
    let n = store.node_count();
    // Go dense once the active set is a large fraction of the graph: past this a
    // dense array's O(1) scatter beats an FnvMap's hashing, and a full-scan seed is
    // dense from the start. Below it the sparse list wins by touching only live nodes.
    let dense_cut = (n / 16).max(1024);
    match plan {
        Plan::Scan { label } => {
            let seed: &[u32] = match label {
                Some(l) => store.nodes_with_label(l),
                None => return Some(dense_from(store.all_nodes().into_iter(), n)),
            };
            if seed.len() > dense_cut {
                let mut counts = vec![0.0f64; n];
                for &v in seed {
                    counts[v as usize] = 1.0;
                }
                Some(Counts::Dense(counts))
            } else {
                Some(Counts::Sparse(seed.iter().map(|&v| (v, 1.0)).collect()))
            }
        }
        Plan::IndexSeek { label, key, value } => Some(sparse_or_dense(
            index_seek_ids(store, label, key, value),
            n,
            dense_cut,
        )),
        Plan::RangeSeek {
            label,
            key,
            op,
            value,
        } => Some(sparse_or_dense(
            range_seek_ids(store, label, key, *op, value),
            n,
            dense_cut,
        )),
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            double_loops,
            ..
        } => {
            if *from + 1 != chain_width(input)? {
                return None;
            }
            let prev = frontier_counts(input, store)?;
            let want = match want_etypes(store, edge_label) {
                Ok(w) => w,
                Err(()) => return Some(Counts::Sparse(Vec::new())), // unknown label → no paths
            };
            // Gremlin `both()` walks a self-loop TWICE — `for_each_nbr` doubles it under
            // `double_loops`, so the multiplicity (hence the count) matches core.
            let dl = *double_loops;
            // Estimate the next frontier's fan-out from the source count and the
            // average degree; go dense when it will be large, sparse otherwise. The
            // scatter itself is identical either way — only the accumulator differs.
            let avg_deg = if n == 0 {
                0.0
            } else {
                store.edge_count() as f64 / n as f64
            };
            let est_next = prev.active() as f64 * avg_deg.max(1.0);
            if est_next > dense_cut as f64 {
                let mut next = vec![0.0f64; n];
                prev.for_each(|v, c| {
                    for_each_nbr(store, v, *dir, &want, dl, |nbr, _| next[nbr as usize] += c);
                });
                Some(Counts::Dense(next))
            } else {
                // Sparse scatter into an FnvMap keyed by neighbour — touches only the
                // few nodes a narrow frontier reaches, no O(node_count) allocation.
                let mut next: FnvMap<u32, f64> = FnvMap::default();
                prev.for_each(|v, c| {
                    for_each_nbr(store, v, *dir, &want, dl, |nbr, _| {
                        *next.entry(nbr).or_insert(0.0) += c;
                    });
                });
                Some(Counts::Sparse(next.into_iter().collect()))
            }
        }
        // A frontier `hasLabel(L…)` filter: drop the count of any node not carrying one
        // of the labels (a bucket binary-search per active node), so a fused count/agg
        // sees through `<hops>.hasLabel(L).count()` instead of materializing.
        Plan::Filter {
            input,
            pred: Expr::IsLabeled { slot, labels },
        } => {
            if *slot + 1 != chain_width(input)? {
                return None; // the filter must test the current frontier
            }
            let counts = frontier_counts(input, store)?;
            // Membership BITSET once (O(total wanted membership)), then O(1) per node —
            // a mid-traversal `hasLabel` over a big count frontier tests up to node_count
            // ids, and a per-node binary_search into the label buckets (is_labeled) is
            // cache-hostile at that scale.
            let mut member = vec![false; store.node_count()];
            for l in labels {
                for &id in store.nodes_with_label(l) {
                    member[id as usize] = true;
                }
            }
            let keep = |id: u32| member[id as usize];
            Some(match counts {
                Counts::Sparse(v) => {
                    Counts::Sparse(v.into_iter().filter(|&(id, _)| keep(id)).collect())
                }
                Counts::Dense(mut a) => {
                    for (i, c) in a.iter_mut().enumerate() {
                        if *c != 0.0 && !keep(i as u32) {
                            *c = 0.0;
                        }
                    }
                    Counts::Dense(a)
                }
            })
        }
        _ => None,
    }
}

/// A DENSE count frontier with each listed id set to 1.0 (for a full unlabeled scan).
fn dense_from(ids: impl Iterator<Item = u32>, n: usize) -> Counts {
    let mut counts = vec![0.0f64; n];
    for v in ids {
        counts[v as usize] = 1.0;
    }
    Counts::Dense(counts)
}

/// Seed a count frontier from a seek's id list: sparse when the result is small
/// (the common selective seek), dense when it is large. Duplicate ids accumulate.
fn sparse_or_dense(ids: Vec<u32>, n: usize, dense_cut: usize) -> Counts {
    if ids.len() > dense_cut {
        let mut counts = vec![0.0f64; n];
        for v in ids {
            counts[v as usize] += 1.0;
        }
        Counts::Dense(counts)
    } else {
        // A seek CAN repeat an id (an index bucket with dups); fold so each node
        // appears once with its multiplicity, matching the dense accumulation.
        let mut map: FnvMap<u32, f64> = FnvMap::default();
        for v in ids {
            *map.entry(v).or_insert(0.0) += 1.0;
        }
        Counts::Sparse(map.into_iter().collect())
    }
}

/// Answer a scalar `count(*)` over `Filter(Scan)` by running only the filter and
/// returning the number of survivors — NO gather of the surviving rows' columns
/// (which the general Filter → Aggregate path builds and immediately discards for a
/// count). Relies on `try_filter_keep`'s vectorized filter; falls back (`None`) when
/// the predicate isn't a fast-path shape or the input isn't `Filter(Scan)`.
fn try_filtered_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None; // count(*) only
    }
    let Plan::Filter { input: scan, pred } = input else {
        return None;
    };
    let Plan::Scan { label } = scan.as_ref() else {
        return None;
    };
    // STREAM a numeric predicate over the label bucket — count matches with raw-f64
    // compares, never materializing the scan's id vector or a keep list. This is
    // core's structure (it iterates the bucket and tests inline) but with the
    // engine's typed compare instead of core's boxed CExpr tree-walk: measured 3.67x
    // core (and ~5x the engine's own materialize-then-filter) on a 200k range count.
    if let Some(c) = try_stream_num_count(store, label, pred) {
        return Some(scalar_num(c as f64));
    }
    if let Some(c) = try_stream_membership_count(store, label, pred) {
        return Some(scalar_num(c as f64));
    }
    if let Some(c) = try_stream_dict_pred_count(store, label, pred) {
        return Some(scalar_num(c as f64));
    }
    // Fallback: materialize the scan and run the general vectorized filter (string /
    // disjunction / NOT predicates the streaming path does not special-case).
    let batch = pull(scan, store, false).ok()?;
    let keep = try_filter_keep(pred, store, &batch)?;
    Some(scalar_num(keep.len() as f64))
}

/// Common gate for the fused single-hop fast paths: `Filter(<pred>, Expand{single-type, Out,
/// from:0, non-bind}(Scan))` over a single-label graph. Returns the scan label, the endpoint
/// edge type, the predicate, and the Expand plan (for `reverse_seed_decide`). `None` otherwise.
fn fused_hop_shape<'a>(
    input: &'a Plan,
    store: &Store,
) -> Option<(&'a Option<String>, u32, &'a Expr, &'a Plan)> {
    let Plan::Filter { input: exp, pred } = input else {
        return None;
    };
    let Plan::Expand {
        from: 0,
        dir: Dir::Out,
        edge_label,
        bind_edge: false,
        double_loops: false,
        ..
    } = exp.as_ref()
    else {
        return None;
    };
    let Plan::Expand { input: scan, .. } = exp.as_ref() else {
        return None;
    };
    let Plan::Scan { label } = scan.as_ref() else {
        return None;
    };
    if store.has_multi_label_edges() {
        return None;
    }
    // A single wanted etype (the per-type CSR keys on one type; a disjunction reorders).
    let want = want_etypes(store, edge_label).ok()?;
    let [w] = want.as_slice() else {
        return None;
    };
    Some((label, *w, pred, exp.as_ref()))
}

/// Visit each type-`w` OUT edge's neighbour whose source is in `label`, in expand order — the
/// flat per-type partition when the source set is the WHOLE graph (an unlabelled scan, or a
/// label every live node carries; 5x fewer touches than a source walk for a sparse type), else
/// each labelled source's per-type slice. One call per (src, nbr) PATH, so multiplicity is
/// preserved. `None` if the per-type CSR overlay is stale (caller falls back to the general
/// path). Shared by the fused count / aggregate / projection.
fn for_each_typed_out(
    store: &Store,
    label: &Option<String>,
    w: u32,
    mut f: impl FnMut(u32),
) -> Option<()> {
    let flat = store.out_typed_flat(w)?; // freshness gate
    let universal = match label {
        None => true,
        Some(l) => store.nodes_with_label(l).len() == store.live_node_count(),
    };
    if universal {
        for a in flat {
            f(a.nbr);
        }
    } else {
        scan_visit(store, label, |i| {
            if let Some(sl) = store.out_typed_csr(i as u32, w) {
                for a in sl {
                    f(a.nbr);
                }
            }
        });
    }
    Some(())
}

/// Fused `count(*)` over `Filter(<numeric conj on the frontier>, Expand(Scan))` — the
/// single-hop analogue of [`try_filtered_count`]'s streaming. Sweep the type's edges (per-type
/// CSR) and test the numeric predicate INLINE against the neighbour's column, counting per
/// (src, nbr) PATH — no `[src, nbr]` batch, no keep list, no separate filter pass. Byte-
/// identical: the same typed compare per path as the materialize path (multiplicity kept,
/// present-null dropped). Declined for a reverse-seedable (selective) predicate — the
/// `reverse_seed_worth` guard prices the reverse walk against the SOURCE scan (node count),
/// which over-fires for a sparse type, so `reverse_seed_decide` returning `Some` here means the
/// endpoint is genuinely more selective than sweeping the type's edges forward.
fn try_fused_hop_num_count(input: &Plan, store: &Store) -> Option<u64> {
    let (label, w, pred, exp) = fused_hop_shape(input, store)?;
    let (key, bounds) = num_conj_on_slot(pred, 1)?;
    let Some(Column::Num { data, present, .. }) = store.column(&key) else {
        return None;
    };
    if reverse_seed_decide(pred, exp, store, false).is_some() {
        return None;
    }
    let mut count = 0u64;
    for_each_typed_out(store, label, w, |nbr| {
        let j = nbr as usize;
        if present[j] && bounds.iter().all(|&(op, t)| num_pred(op, data[j], t)) {
            count += 1;
        }
    })?;
    Some(count)
}

/// Fused numeric-filtered PROJECTION over `Project(Filter(<numeric conj>, Expand(Scan)))`.
/// The general path pulls the whole expand into an `[src, nbr]` batch, filters, and GATHERS
/// the survivors — a fixed ~0.8ms for an 80k-edge hop no matter how few rows survive, which
/// loses to core's streaming when the filter is mid-selective (survivors ≪ edges) but not
/// selective enough to reverse-seed. Instead STREAM the type's edges (per-type CSR), test the
/// numeric predicate inline, collect just the surviving TARGET ids, and evaluate the projection
/// over that survivor frontier — the survivor count of output rows, never the `[src, nbr]`
/// intermediate. Byte-identical: same typed test per (src, nbr) PATH, survivors in the same
/// (source, out_adj) order the expand emits, projection unchanged. `None` unless every projected
/// item reads only the endpoint (slot 1), lineage-free, single-type Out hop, per-type CSR fresh,
/// and the endpoint is not reverse-seedable.
fn try_fused_hop_project(
    input: &Plan,
    items: &[(String, Expr)],
    store: &Store,
    track: bool,
) -> Option<Batch> {
    if track || items.is_empty() {
        return None;
    }
    let (label, w, pred, exp) = fused_hop_shape(input, store)?;
    if !items.iter().all(|(_, e)| refs_only_slot(e, 1)) {
        return None;
    }
    let (key, bounds) = num_conj_on_slot(pred, 1)?;
    let Some(Column::Num { data, present, .. }) = store.column(&key) else {
        return None;
    };
    if reverse_seed_decide(pred, exp, store, false).is_some() {
        return None;
    }
    // Stream the type's edges, collecting just the surviving TARGET ids (output-proportional,
    // never the `[src, nbr]` intermediate); then evaluate the projection over that frontier.
    let mut survivors: Vec<u32> = Vec::new();
    for_each_typed_out(store, label, w, |nbr| {
        let j = nbr as usize;
        if present[j] && bounds.iter().all(|&(op, t)| num_pred(op, data[j], t)) {
            survivors.push(nbr);
        }
    })?;
    // Evaluate the projection over the survivor frontier (endpoint at slot 1).
    let cols = vec![
        Col::Nodes(vec![0u32; survivors.len()]),
        Col::Nodes(survivors),
    ];
    let out = eval_all(items.iter().map(|(_, e)| e), store, &Batch::of(cols)).ok()?;
    Some(Batch::of(out))
}

/// Fused scalar aggregate — `count(*)` / `sum` / `min` / `max` — over
/// `Filter(<any endpoint-only pred>, Expand(Scan))` for a predicate the inline numeric count
/// can't take (an OR, a mixed-key disjunction, a string search). Sweep the type's edges off
/// the per-type CSR into just the TARGET-id column, run the SAME vectorized `eval_mask` the
/// materialize path would, and fold the aggregate over the TRUE cells — skipping the
/// `[src, nbr]` batch the general Aggregate builds AND the keep-gather it discards. Byte-
/// identical: the mask is per (src, nbr) PATH exactly as the materialize filter, and the flat
/// partition is in the SAME (source, out_adj) order the expand emits, so a float `sum` folds in
/// the identical order. Declined for a reverse-seedable (selective) endpoint.
fn try_fused_hop_mask_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.distinct {
        return None;
    }
    // count(*) has no arg; sum/min/max fold a Num property of the frontier (slot 1).
    let arg_key: Option<&String> = match (&agg.func, agg.arg.as_ref()) {
        (AggFn::Count, None) => None,
        (AggFn::Sum | AggFn::Min | AggFn::Max, Some(Expr::Prop { slot: 1, key })) => Some(key),
        _ => return None,
    };
    let (label, w, pred, exp) = fused_hop_shape(input, store)?;
    if !refs_only_slot(pred, 1) {
        return None;
    }
    // The agg property must be a plain Num column (min/max/sum semantics; a NULL cell is
    // skipped, matching the general aggregate).
    let agg_col: Option<(&[f64], &[bool])> = match arg_key {
        Some(k) => match store.column(k)? {
            Column::Num { data, present, .. } => Some((data, present)),
            _ => return None,
        },
        None => None,
    };
    if reverse_seed_decide(pred, exp, store, false).is_some() {
        return None;
    }
    let mut targets: Vec<u32> = Vec::new();
    for_each_typed_out(store, label, w, |nbr| targets.push(nbr))?;
    // Frontier at slot 1; slot 0 is a dummy the endpoint-only predicate never reads.
    let cols = vec![Col::Nodes(vec![0u32; targets.len()]), Col::Nodes(targets)];
    let batch = Batch::of(cols);
    let mask = eval_mask(pred, store, &batch).ok()?;
    let Col::Nodes(targets) = batch.slot(1) else {
        return None;
    };
    let is_true = |i: usize| mask.get(i) == Some(&Some(true));
    match (&agg.func, agg_col) {
        (AggFn::Count, _) => {
            let c = (0..targets.len()).filter(|&i| is_true(i)).count();
            Some(scalar_num(c as f64))
        }
        (AggFn::Sum, Some((data, present))) => {
            let mut total = 0f64;
            for (i, &t) in targets.iter().enumerate() {
                let j = t as usize;
                if is_true(i) && present[j] {
                    total += data[j];
                }
            }
            Some(scalar_num(total))
        }
        (AggFn::Min | AggFn::Max, Some((data, present))) => {
            let want_min = matches!(agg.func, AggFn::Min);
            let mut best: Option<f64> = None;
            for (i, &t) in targets.iter().enumerate() {
                let j = t as usize;
                if is_true(i) && present[j] {
                    let x = data[j];
                    best = Some(match best {
                        None => x,
                        Some(b) => {
                            let ord = value::cmp_num_total(x, b);
                            if (want_min && ord.is_lt()) || (!want_min && ord.is_gt()) {
                                x
                            } else {
                                b
                            }
                        }
                    });
                }
            }
            Some(Batch::single(Col::Gen(vec![
                best.map_or(Value::Null, Value::Num)
            ])))
        }
        _ => None,
    }
}

/// Count nodes of `label` whose `pred` holds, STREAMING the label bucket with raw
/// f64 compares — no scan-id materialization, no keep vector. Handles a single
/// `prop OP num` compare and a same-column numeric range (`lo <= x AND x < hi`), the
/// hot filtered-count shapes; `None` for anything else (the caller materializes and
/// runs the general filter). Every survivor test matches `try_filter_keep`'s typed
/// paths exactly (present gates NULL; a NaN cell fails ordering → dropped), so the
/// count is identical.
/// Recognize a filter predicate that is a CONJUNCTION of numeric compares all on the
/// SAME property of one `slot` — `prop OP num` (either operand order) — returning
/// `(key, bounds)`. Shared by the streaming node/edge count fast paths; `None` for a
/// string / disjunction / multi-slot / multi-key / non-numeric predicate.
fn num_conj_on_slot(pred: &Expr, slot: usize) -> Option<(String, Vec<(CompareOp, f64)>)> {
    // An atom on `slot` is either a numeric compare (a bound) or a `PropertyExists`
    // presence gate (NO bound — redundant with the streaming count's own `present[i]`
    // check, and implied by any compare on the same key). `has(k, pred)` desugars to
    // `And(PropertyExists{k}, <compare>)`, so accepting the presence atom is what keeps
    // a non-selective `has('age', neq(60)).count()` on the streaming path instead of
    // materializing a 99% keep-list.
    let atom = |e: &Expr| -> Option<(String, Option<(CompareOp, f64)>)> {
        match e {
            Expr::PropertyExists { slot: s, key } if *s == slot => Some((key.clone(), None)),
            Expr::Compare { op, left, right } => {
                let (key, op, lit) = match (left.as_ref(), right.as_ref()) {
                    (Expr::Prop { slot: s, key }, Expr::Lit(v)) if *s == slot => {
                        (key.clone(), *op, v)
                    }
                    (Expr::Lit(v), Expr::Prop { slot: s, key }) if *s == slot => {
                        (key.clone(), flip_op(*op), v)
                    }
                    _ => return None,
                };
                match lit {
                    Value::Num(t) => Some((key, Some((op, *t)))),
                    _ => None,
                }
            }
            _ => None,
        }
    };
    let mut conjuncts = Vec::new();
    flatten_and(pred, &mut conjuncts);
    let mut key0: Option<String> = None;
    let mut bounds: Vec<(CompareOp, f64)> = Vec::with_capacity(conjuncts.len());
    for c in &conjuncts {
        let (key, bound) = atom(c)?;
        match &key0 {
            Some(k) if *k != key => return None, // a second key can't stream one column
            _ => key0 = Some(key),
        }
        if let Some(b) = bound {
            bounds.push(b);
        }
    }
    Some((key0?, bounds))
}

/// Answer `count(*)` over `Filter(edge-pred, Expand{bind_edge})` by STREAMING the
/// expansion — for each source, test each matching out-edge's property inline and
/// count — instead of materializing every `(source, edge, target)` row and filtering
/// (an O(edges) Batch). Edge properties are boxed (a per-key eid→Value map), so the
/// per-edge lookup stays, but the row materialization is what dominated. The survivor
/// test matches the general Filter (a present Num edge prop tests the bounds;
/// null/non-numeric → UNKNOWN → dropped), so the count is identical. Only the pred on
/// the bound EDGE slot (not the target node) is handled; anything else falls through.
fn try_edge_filtered_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None; // count(*) only
    }
    let Plan::Filter {
        input: expand,
        pred,
    } = input
    else {
        return None;
    };
    let Plan::Expand {
        input: src,
        from,
        dir,
        edge_label,
        bind_edge,
        double_loops: false,
    } = expand.as_ref()
    else {
        return None;
    };
    if !bind_edge {
        return None; // the edge must be bound for the filter to read its property
    }
    // A bind_edge Expand appends the edge at the slot just past its input (then the
    // target node); the pred must be a numeric conjunction on that edge slot.
    let edge_slot = from + 1;
    let (key, bounds) = num_conj_on_slot(pred, edge_slot)?;
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)), // unknown edge type → no rows
    };
    let src_ids = frontier_ids(src, store)?;
    let mut count = 0u64;
    // Typed overlay: read the edge property as a raw f64 (no per-edge hash probe +
    // Value unbox). Falls back to the boxed edge_prop when the overlay is stale or the
    // key is not homogeneously numeric.
    if let Some((data, present)) = store.edge_num_column(&key) {
        for &v in &src_ids {
            for_each_nbr(store, v, *dir, &want, false, |_nbr, eid| {
                let i = eid as usize;
                if present[i] && bounds.iter().all(|&(op, t)| num_pred(op, data[i], t)) {
                    count += 1;
                }
            });
        }
    } else {
        for &v in &src_ids {
            for_each_nbr(store, v, *dir, &want, false, |_nbr, eid| {
                if let Value::Num(x) = store.edge_prop(eid, &key) {
                    if bounds.iter().all(|&(op, t)| num_pred(op, x, t)) {
                        count += 1;
                    }
                }
            });
        }
    }
    Some(scalar_num(count as f64))
}

/// A searched `CASE` that is a categorical remap: EVERY branch condition is
/// `<dict col> = <string literal>` on ONE key. Returns the slot, key, and a code →
/// first-matching-branch-index table (`None` where no branch matches that dict value).
fn case_dict_lookup(
    branches: &[(Expr, Expr)],
    store: &Store,
) -> Option<(usize, String, Vec<Option<usize>>)> {
    if branches.is_empty() {
        return None;
    }
    let mut slot_key: Option<(usize, String)> = None;
    let mut lits: Vec<&str> = Vec::with_capacity(branches.len());
    for (cond, _) in branches {
        let Expr::Compare {
            op: CompareOp::Eq,
            left,
            right,
        } = cond
        else {
            return None;
        };
        let (s, k, v) = match (left.as_ref(), right.as_ref()) {
            (Expr::Prop { slot, key }, Expr::Lit(Value::Str(v)))
            | (Expr::Lit(Value::Str(v)), Expr::Prop { slot, key }) => (*slot, key, v),
            _ => return None,
        };
        match &slot_key {
            None => slot_key = Some((s, k.clone())),
            Some((s0, k0)) if *s0 == s && k0 == k => {}
            Some(_) => return None,
        }
        lits.push(v.as_ref());
    }
    let (slot, key) = slot_key?;
    let Some(Column::Dict { dict, .. }) = store.column(&key) else {
        return None;
    };
    let code_to_branch: Vec<Option<usize>> = dict
        .iter()
        .map(|dstr| lits.iter().position(|lit| dstr.as_ref() == *lit))
        .collect();
    Some((slot, key, code_to_branch))
}

/// Is `pred` a disjunction (or a single term) of `slot.key == <literal>`, all on ONE
/// key? Returns that key and the literal values — the shape `x IN [a, b, …]` desugars to.
fn eq_disjunction_on_slot(pred: &Expr, slot: usize) -> Option<(String, Vec<Value>)> {
    fn collect(e: &Expr, slot: usize, key: &mut Option<String>, vals: &mut Vec<Value>) -> bool {
        match e {
            Expr::Or(a, b) => collect(a, slot, key, vals) && collect(b, slot, key, vals),
            Expr::Compare {
                op: CompareOp::Eq,
                left,
                right,
            } => {
                let (k, v) = match (left.as_ref(), right.as_ref()) {
                    (Expr::Prop { slot: s, key }, Expr::Lit(v))
                    | (Expr::Lit(v), Expr::Prop { slot: s, key })
                        if *s == slot =>
                    {
                        (key, v)
                    }
                    _ => return false,
                };
                if v.is_null() {
                    return false; // a NULL term makes non-matches UNKNOWN, not FALSE
                }
                match key.as_deref() {
                    None => *key = Some(k.clone()),
                    Some(existing) if existing == k => {}
                    Some(_) => return false, // mixed keys — not this shape
                }
                vals.push(v.clone());
                true
            }
            _ => false,
        }
    }
    let (mut key, mut vals) = (None, Vec::new());
    collect(pred, slot, &mut key, &mut vals).then_some(())?;
    Some((key?, vals))
}

/// If every leaf of `pred` that touches a row is `Prop { slot, key }` for ONE `(slot,
/// key)` (no label tests, presence tests, or other columns), return it — the predicate is
/// a pure function of a single property, evaluable once per distinct value.
fn sole_prop_ref(pred: &Expr) -> Option<(usize, String)> {
    fn walk(e: &Expr, seen: &mut Option<(usize, String)>) -> bool {
        match e {
            Expr::Lit(_) => true,
            Expr::Prop { slot, key } => match seen {
                None => {
                    *seen = Some((*slot, key.clone()));
                    true
                }
                Some((s0, k0)) => *s0 == *slot && k0 == key,
            },
            Expr::Not(a) => walk(a, seen),
            Expr::And(a, b) | Expr::Or(a, b) | Expr::Xor(a, b) => walk(a, seen) && walk(b, seen),
            Expr::Compare { left, right, .. } => walk(left, seen) && walk(right, seen),
            Expr::Call { args, .. } => args.iter().all(|a| walk(a, seen)),
            Expr::In { needle, haystack } => walk(needle, seen) && walk(haystack, seen),
            _ => false, // other slots, subqueries, label/presence tests → not pure-in-key
        }
    }
    let mut seen = None;
    walk(pred, &mut seen).then_some(())?;
    seen
}

/// [`sole_prop_ref`] constrained to a specific `slot` (returns just the key).
fn sole_prop_key(pred: &Expr, slot: usize) -> Option<String> {
    sole_prop_ref(pred)
        .filter(|(s, _)| *s == slot)
        .map(|(_, k)| k)
}

/// Evaluate a single-property predicate for one concrete value of that property, by
/// substituting the literal and folding the now-constant expression. `None` on a faulting
/// eval; `Some(true)` only when the result is definitely TRUE (3VL — the keep condition).
fn dict_pred_value(pred: &Expr, slot: usize, key: &str, v: &Value, store: &Store) -> Option<bool> {
    let e = subst_prop(pred, slot, key, v);
    let col = eval(&e, store, &Batch::single(Col::Num(vec![0.0]))).ok()?;
    Some(matches!(col.value_at(0), Value::Bool(true)))
}

/// Evaluate a scalar expression that is a pure function of one DICT column by computing it
/// once per distinct dict value (≤ dict size) and mapping each row to the shared result —
/// so `upper(city)`, `substring(city, …)`, etc. over a categorical column do dict.len()
/// string allocations instead of one per row (the result `Value`s, including `Arc<str>`,
/// are cloned per row = a refcount bump, not a new allocation). Byte-identical: the value
/// is a function of the property alone (absent → the NULL case, computed once).
/// Fold the boxed `Vec<Value>` a scalar function produced into a TYPED column when every
/// cell is the same non-null primitive (`Num`/`Str`/`Bool`). A downstream sort, DISTINCT or
/// GROUP BY on the computed value then takes the typed fast path (raw f64 / `Arc<str>`
/// compare, dict-code dedup) instead of boxing every cell per comparison. A single null or a
/// mixed type keeps it `Gen`; either way `value_at(i)` is byte-identical for every row, so
/// this is purely an internal representation choice — never an observable one.
fn typed_col_from_values(out: Vec<Value>) -> Col {
    let Some(first) = out.first() else {
        return Col::Gen(out);
    };
    match first {
        Value::Num(_) if out.iter().all(|v| matches!(v, Value::Num(_))) => Col::Num(
            out.iter()
                .map(|v| {
                    if let Value::Num(x) = v {
                        *x
                    } else {
                        unreachable!()
                    }
                })
                .collect(),
        ),
        Value::Str(_) if out.iter().all(|v| matches!(v, Value::Str(_))) => Col::Str(
            out.into_iter()
                .map(|v| {
                    if let Value::Str(s) = v {
                        s
                    } else {
                        unreachable!()
                    }
                })
                .collect(),
        ),
        Value::Bool(_) if out.iter().all(|v| matches!(v, Value::Bool(_))) => Col::Bool(
            out.iter()
                .map(|v| {
                    if let Value::Bool(b) = v {
                        *b
                    } else {
                        unreachable!()
                    }
                })
                .collect(),
        ),
        _ => Col::Gen(out),
    }
}

fn try_eval_dict_scalar(expr: &Expr, store: &Store, batch: &Batch) -> Option<Col> {
    let (slot, key) = sole_prop_ref(expr)?;
    let Col::Nodes(ids) = batch.slot(slot) else {
        return None;
    };
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(&key)
    else {
        return None;
    };
    let ev = |v: &Value| -> Option<Value> {
        let e = subst_prop(expr, slot, &key, v);
        Some(
            eval(&e, store, &Batch::single(Col::Num(vec![0.0])))
                .ok()?
                .value_at(0),
        )
    };
    let mut per_code = Vec::with_capacity(dict.len());
    for dv in dict.iter() {
        per_code.push(ev(&Value::Str(dv.clone()))?);
    }
    let null_val = ev(&Value::Null)?;
    Some(Col::Gen(
        ids.iter()
            .map(|&id| {
                if id != u32::MAX && present[id as usize] {
                    per_code[codes[id as usize] as usize].clone()
                } else {
                    null_val.clone()
                }
            })
            .collect(),
    ))
}

/// Keep-list for a predicate that is a pure function of one DICT column: evaluate it once
/// per distinct dict value, then keep rows whose code matches — the projection sibling of
/// [`try_stream_dict_pred_count`], for `WHERE <dict pred> RETURN …`. Byte-identical to the
/// per-row boxed filter (same 3VL TRUE-keeps rule).
fn try_filter_keep_dict(pred: &Expr, store: &Store, batch: &Batch) -> Option<Vec<usize>> {
    let (slot, key) = sole_prop_ref(pred)?;
    let Col::Nodes(ids) = batch.slot(slot) else {
        return None;
    };
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(&key)
    else {
        return None;
    };
    let mut matches = Vec::with_capacity(dict.len());
    for dv in dict.iter() {
        matches.push(dict_pred_value(
            pred,
            slot,
            &key,
            &Value::Str(dv.clone()),
            store,
        )?);
    }
    let null_match = dict_pred_value(pred, slot, &key, &Value::Null, store)?;
    Some(
        ids.iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                let hit = if id != u32::MAX && present[id as usize] {
                    matches[codes[id as usize] as usize]
                } else {
                    null_match
                };
                hit.then_some(i)
            })
            .collect(),
    )
}

/// Replace every `Prop { slot, key }` in `e` with `Lit(value)` — so a predicate that is a
/// pure function of one property becomes a constant expression, evaluable once.
fn subst_prop(e: &Expr, slot: usize, key: &str, value: &Value) -> Expr {
    match e {
        Expr::Prop { slot: s, key: k } if *s == slot && k == key => Expr::Lit(value.clone()),
        Expr::Not(a) => Expr::Not(Box::new(subst_prop(a, slot, key, value))),
        Expr::And(a, b) => Expr::And(
            Box::new(subst_prop(a, slot, key, value)),
            Box::new(subst_prop(b, slot, key, value)),
        ),
        Expr::Or(a, b) => Expr::Or(
            Box::new(subst_prop(a, slot, key, value)),
            Box::new(subst_prop(b, slot, key, value)),
        ),
        Expr::Xor(a, b) => Expr::Xor(
            Box::new(subst_prop(a, slot, key, value)),
            Box::new(subst_prop(b, slot, key, value)),
        ),
        Expr::Compare { op, left, right } => Expr::Compare {
            op: *op,
            left: Box::new(subst_prop(left, slot, key, value)),
            right: Box::new(subst_prop(right, slot, key, value)),
        },
        Expr::Call { name, args } => Expr::Call {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| subst_prop(a, slot, key, value))
                .collect(),
        },
        Expr::In { needle, haystack } => Expr::In {
            needle: Box::new(subst_prop(needle, slot, key, value)),
            haystack: Box::new(subst_prop(haystack, slot, key, value)),
        },
        other => other.clone(),
    }
}

/// Streaming count for ANY predicate that is a pure function of one DICT column: evaluate
/// it once per distinct dict value (≤ dict size), then count code membership in a single
/// pass — never materializing rows. Covers `STARTS WITH … OR …`, `CONTAINS`, ranges, and
/// arbitrary boolean combinations over a categorical column. Byte-identical: a row is
/// counted iff the predicate is definitely TRUE for its value (or NULL, evaluated once).
fn try_stream_dict_pred_count(store: &Store, label: &Option<String>, pred: &Expr) -> Option<u64> {
    let key = sole_prop_key(pred, 0)?;
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(&key)
    else {
        return None;
    };
    let eval_const = |v: &Value| -> Option<bool> {
        let e = subst_prop(pred, 0, &key, v);
        let col = eval(&e, store, &Batch::single(Col::Num(vec![0.0]))).ok()?;
        Some(matches!(col.value_at(0), Value::Bool(true)))
    };
    let mut matches = Vec::with_capacity(dict.len());
    for dv in dict.iter() {
        matches.push(eval_const(&Value::Str(dv.clone()))?);
    }
    let null_match = eval_const(&Value::Null)?;
    let mut count = 0u64;
    scan_visit(store, label, |i| {
        let hit = if present[i] {
            matches[codes[i] as usize]
        } else {
            null_match
        };
        if hit {
            count += 1;
        }
    });
    Some(count)
}

/// Streaming count for categorical membership — `col IN [a, b, …]` (desugared to an
/// OR-chain of equals) on a `Dict` or `Str` column — the string sibling of
/// `try_stream_num_count`. Maps the literals to dict CODES once, then counts matches in
/// one pass over the bucket, never materializing an id vector or keep list. Byte-identical
/// to the OR filter (a literal absent from the dict simply matches nothing).
fn try_stream_membership_count(store: &Store, label: &Option<String>, pred: &Expr) -> Option<u64> {
    let (key, vals) = eq_disjunction_on_slot(pred, 0)?;
    let mut count = 0u64;
    match store.column(&key)? {
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            // Every literal must be a string (else it can't equal a dict value). Resolve
            // to codes; a literal not in the dict contributes no code (never matches).
            let mut targets: Vec<u32> = Vec::with_capacity(vals.len());
            for v in &vals {
                let Value::Str(s) = v else { return None };
                if let Some(c) = dict.iter().position(|d| d.as_ref() == s.as_ref()) {
                    targets.push(c as u32);
                }
            }
            scan_visit(store, label, |i| {
                if present[i] && targets.contains(&codes[i]) {
                    count += 1;
                }
            });
        }
        Column::Str { data, present, .. } => {
            let mut targets: Vec<&str> = Vec::with_capacity(vals.len());
            for v in &vals {
                let Value::Str(s) = v else { return None };
                targets.push(s.as_ref());
            }
            scan_visit(store, label, |i| {
                if present[i] && targets.iter().any(|t| data[i].as_ref() == *t) {
                    count += 1;
                }
            });
        }
        _ => return None,
    }
    Some(count)
}

/// `DISTINCT`/`dedup()` over `values(<dict col>)`: emit the distinct dict values in
/// FIRST-SEEN order by scanning codes against a `dict.len()` bitset — never decoding or
/// hashing the per-row strings. Byte-identical to the general first-seen dedup (first
/// occurrence of a code == first occurrence of its string). Bails (→ general path) on any
/// absent value or null-sentinel id, whose NULL dedup this fast path doesn't model.
fn try_distinct_dict_col(input: &Plan, store: &Store) -> Option<Batch> {
    let Plan::Project { input: pin, items } = input else {
        return None;
    };
    let [(_, Expr::Prop { slot, key })] = items.as_slice() else {
        return None;
    };
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(key)
    else {
        return None;
    };
    let frontier = pull(pin, store, false).ok()?;
    // The property may sit on any bound slot — slot 0 for `values(k).dedup()`, but the
    // hop endpoint (e.g. slot 2) for `out().out().values(k).dedup()`.
    let Col::Nodes(ids) = frontier.slot(*slot) else {
        return None;
    };
    let mut seen = vec![false; dict.len()];
    let mut out: Vec<Arc<str>> = Vec::new();
    for &id in ids {
        if id == u32::MAX {
            return None; // a NULL value in the dedup — let the general path handle it
        }
        let i = id as usize;
        if !present[i] {
            return None;
        }
        let c = codes[i] as usize;
        if !seen[c] {
            seen[c] = true;
            out.push(dict[c].clone());
        }
    }
    Some(Batch::of(vec![Col::Str(out)]))
}

/// `MATCH (a)-[]->(b) WHERE b.k <op> a.k RETURN count(*)` — a hop whose survival compares
/// the two ENDPOINTS' numeric properties. Stream the edges and compare the source/neighbor
/// num columns directly, never building the neighbor frontier or a boxed compare. A count
/// is order-free, so byte-identical (NaN / absent → not counted, matching the 3VL filter).
fn try_edge_cross_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None;
    }
    let Plan::Filter {
        input: expand,
        pred,
    } = input
    else {
        return None;
    };
    let Plan::Expand {
        input: scan,
        from: 0, // source at slot 0, neighbour appended at slot 1
        dir,
        edge_label,
        bind_edge: false,
        double_loops: _,
    } = expand.as_ref()
    else {
        return None;
    };
    let Plan::Scan { label } = scan.as_ref() else {
        return None;
    };
    // The predicate is the endpoint compare, optionally AND a neighbour-label test (the
    // `(b:Label)` pattern) — peel that off and check the label per edge.
    let (cmp, nbr_labels): (&Expr, Option<&[String]>) = match pred {
        Expr::Compare { .. } => (pred, None),
        Expr::And(a, b) => match (a.as_ref(), b.as_ref()) {
            (c @ Expr::Compare { .. }, Expr::IsLabeled { slot: 1, labels })
            | (Expr::IsLabeled { slot: 1, labels }, c @ Expr::Compare { .. }) => (c, Some(labels)),
            _ => return None,
        },
        _ => return None,
    };
    // The compare relates two properties, each on slot 0 (source) or slot 1 (neighbour).
    let Expr::Compare { op, left, right } = cmp else {
        return None;
    };
    let (Expr::Prop { slot: ls, key: lk }, Expr::Prop { slot: rs, key: rk }) =
        (left.as_ref(), right.as_ref())
    else {
        return None;
    };
    if *ls > 1 || *rs > 1 {
        return None;
    }
    let (
        Some(Column::Num {
            data: ld,
            present: lp,
            ..
        }),
        Some(Column::Num {
            data: rd,
            present: rp,
            ..
        }),
    ) = (store.column(lk), store.column(rk))
    else {
        return None; // unboxed numeric endpoints only
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)),
    };
    // O(1) neighbour-label check: a membership bitset beats a `is_labeled` binary-search
    // per edge (which made this LOSE to the general path).
    let nbr_bits: Option<Vec<bool>> = nbr_labels.map(|labels| {
        let mut b = vec![false; store.node_count()];
        for l in labels {
            for &id in store.nodes_with_label(l) {
                b[id as usize] = true;
            }
        }
        b
    });
    let mut count = 0u64;
    scan_visit(store, label, |src| {
        let src = src as u32;
        for_each_nbr(store, src, *dir, &want, false, |nbr, _| {
            if let Some(bits) = &nbr_bits {
                if !bits[nbr as usize] {
                    return; // neighbour fails its `(b:Label)` constraint
                }
            }
            let li = if *ls == 0 { src } else { nbr } as usize;
            let ri = if *rs == 0 { src } else { nbr } as usize;
            if lp[li] && rp[ri] && num_pred(*op, ld[li], rd[ri]) {
                count += 1;
            }
        });
    });
    Some(scalar_num(count as f64))
}

fn try_stream_num_count(store: &Store, label: &Option<String>, pred: &Expr) -> Option<u64> {
    let (key, bounds) = num_conj_on_slot(pred, 0)?;
    let Some(Column::Num {
        data,
        present,
        nulls,
    }) = store.column(&key)
    else {
        return None;
    };
    let mut count = 0u64;
    scan_visit(store, label, |i| {
        // A bare presence gate (`has(k)`, no bounds) counts PRESENCE — a stored
        // present-null included (`present || nulls`). A bounded compare counts only
        // typed values (`present`): a null satisfies no numeric predicate.
        if bounds.is_empty() {
            if present[i] || nulls[i] {
                count += 1;
            }
        } else if present[i] && bounds.iter().all(|&(op, t)| num_pred(op, data[i], t)) {
            count += 1;
        }
    });
    Some(count)
}

/// Answer a scalar `count(*)` over a `VarLength` hop by DFS-counting the emitted
/// paths per source row, WITHOUT materializing the (up to millions of) keep/ends
/// vectors or gathering the input slots — which the general VarLength → Aggregate
/// path builds and immediately discards for a count. Same traversal, edge-type
/// filter and trail bookkeeping as `var_length`, so the count is exact and
/// identical. `None` for a grouped / arg'd / DISTINCT aggregate or a non-`VarLength`
/// input (handled elsewhere).
fn try_varlen_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None; // count(*) only
    }
    let Plan::VarLength {
        input: inner,
        from,
        dir,
        edge_label,
        min,
        max,
        mode,
        until,
        body_filter,
        double_loops,
    } = input
    else {
        return None;
    };
    if until.is_some() || body_filter.is_some() {
        return None; // an until(pred) walk emits a filtered subset — no closed-form count
    }
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)), // unknown edge type → no paths
    };
    let batch = pull(inner, store, false).ok()?;
    let Col::Nodes(src) = batch.slot(*from) else {
        return None;
    };

    // ALGEBRAIC count: for a bounded OUT walk/trail with max<=2, count(*) is the sum of
    // per-hop path counts computed from degrees in O(V+E) — NOT by enumerating the
    // O(paths) walks. 1-hop = the source out-edges; 2-hop = for each source out-edge
    // s->y, the neighbour's out-degree. A TRAIL (no edge reuse) then excludes the one
    // reused-self-loop path s->s->s over the same edge; a WALK (repeat()'s default)
    // permits it, so it makes no correction. Only taken when the enumeration would be
    // the MORE expensive path (a large source set); a filtered / small source stays on
    // the DFS below, where enumeration is already cheap. WALK/TRAIL only: the degree
    // algebra counts node-repeating paths, which SIMPLE / ACYCLIC forbid — those must
    // enumerate via the DFS below.
    let is_trail = matches!(mode, PathMode::Trail);
    if matches!(dir, Dir::Out)
        && *max <= 2
        && *max >= 1
        && matches!(mode, PathMode::Trail | PathMode::Walk)
    {
        let (nc, ec) = (store.node_count(), store.edge_count());
        let avg_deg = if nc == 0 { 0.0 } else { ec as f64 / nc as f64 };
        let est_paths = src.len() as f64 * avg_deg.powi(*max as i32);
        if est_paths > 2.0 * (nc + ec) as f64 {
            let mut outdeg = vec![0u64; nc];
            for (v, d) in outdeg.iter_mut().enumerate() {
                *d = if want.is_empty() {
                    store.out(v as u32).len() as u64
                } else {
                    store
                        .out(v as u32)
                        .iter()
                        .filter(|a| edge_carries_wanted(store, a, &want))
                        .count() as u64
                };
            }
            let mut total: u64 = 0;
            for &s in src {
                for a in store.out(s) {
                    if !edge_carries_wanted(store, a, &want) {
                        continue;
                    }
                    if *min <= 1 {
                        total += 1; // the 1-hop path s -> a.nbr
                    }
                    if *max >= 2 {
                        total += outdeg[a.nbr as usize]; // 2-hop paths s -> a.nbr -> z
                        if is_trail && a.nbr == s {
                            total -= 1; // a trail excludes the reused self-loop s -> s -> s
                        }
                    }
                }
            }
            return Some(scalar_num(total as f64));
        }
    }

    let mut total: u64 = 0;
    let mut used: Vec<u32> = Vec::new();
    let node_unique = matches!(mode, PathMode::Simple | PathMode::Acyclic);
    for &v in src {
        if node_unique {
            used.push(v); // mark the start node
        }
        varlen_count_dfs(
            store,
            v,
            0,
            *min,
            *max,
            *dir,
            &want,
            *mode,
            v,
            &mut used,
            &mut total,
            *double_loops,
        );
        if node_unique {
            used.pop();
        }
        debug_assert!(used.is_empty());
    }
    Some(scalar_num(total as f64))
}

/// The shared LEAN iterative walker behind the count/agg var-length fast-paths: like
/// `varlen_walk` but with neither the path stacks nor an emit sink — it just calls
/// `visit(v)` once per "row" the materializing path would emit (every length in
/// `min..=max`, plus each `Close` endpoint). An explicit heap frame stack, so it uses
/// O(1) CALL stack however deep the closure — the recursive twins it replaced went one
/// frame per hop, so a deep count/agg could overflow (or commit the 1 GiB big stack).
#[allow(clippy::too_many_arguments)]
fn varlen_scan_walk(
    store: &Store,
    v0: u32,
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    mode: PathMode,
    start: u32,
    used: &mut Vec<u32>,
    double_loops: bool,
    visit: &mut dyn FnMut(u32),
) {
    // Root pre-work: the length-0 source is a row iff `min == 0`; no descent past `max`.
    if min == 0 {
        visit(v0);
    }
    if max == 0 {
        return;
    }
    let drop_loop = matches!(dir, Dir::Both) && !double_loops;
    struct SF {
        v: u32,
        len: u32,
        cursor: usize,
        pending: Option<bool>, // Some(pop_used) once we've descended into a child
    }
    let mut stack = vec![SF {
        v: v0,
        len: 0,
        cursor: 0,
        pending: None,
    }];
    'frames: while let Some(top) = stack.last_mut() {
        if let Some(pop_used) = top.pending.take() {
            if pop_used {
                used.pop();
            }
        }
        let (v, len) = (top.v, top.len);
        loop {
            let Some((is_inc, a)) = adj_nth(store, v, dir, top.cursor) else {
                stack.pop();
                continue 'frames;
            };
            top.cursor += 1;
            if !want.is_empty()
                && !want.iter().any(|&w| {
                    w == a.etype
                        || (store.has_multi_label_edges() && store.edge_has_label(a.eid, w))
                })
            {
                continue;
            }
            if is_inc && drop_loop && a.nbr == v {
                continue;
            }
            let mark = match varlen_step(mode, start, &a, used) {
                VarStep::Skip => continue,
                VarStep::Close => {
                    if len + 1 >= min {
                        visit(a.nbr);
                    }
                    continue;
                }
                VarStep::Go(mark) => mark,
            };
            // Child pre-work at len+1: it is a row iff in range, then descend unless at
            // `max` (the recursion's `len == max` early return — its mark push/pop around
            // an immediately-returning child cannot affect the tally, so we skip it).
            let clen = len + 1;
            if clen >= min {
                visit(a.nbr);
            }
            if clen == max {
                continue;
            }
            if let Some(m) = mark {
                used.push(m);
            }
            top.pending = Some(mark.is_some());
            stack.push(SF {
                v: a.nbr,
                len: clen,
                cursor: 0,
                pending: None,
            });
            continue 'frames;
        }
    }
}

/// The counting twin of `varlen_dfs`: tallies every row the materializing path would
/// emit. Iterative (see [`varlen_scan_walk`]) so a deep count can't overflow the stack.
#[allow(clippy::too_many_arguments)]
fn varlen_count_dfs(
    store: &Store,
    v: u32,
    _len: u32, // always 0 at the call sites — the walk starts at the source
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    mode: PathMode,
    start: u32,
    used: &mut Vec<u32>,
    total: &mut u64,
    double_loops: bool,
) {
    varlen_scan_walk(
        store,
        v,
        min,
        max,
        dir,
        want,
        mode,
        start,
        used,
        double_loops,
        &mut |_| {
            *total += 1;
        },
    );
}

/// The outcome of the per-hop reuse gate ([`varlen_step`]).
enum VarStep {
    /// The hop is forbidden — skip this neighbour.
    Skip,
    /// A SIMPLE closing hop (`nbr == start`): emit the endpoint but do NOT descend —
    /// the cycle is closed, and extending it would repeat an interior node (mirrors
    /// core's `is_close` early-`continue`).
    Close,
    /// Descend. `Some(id)` is pushed onto the reuse stack before recursing (Trail:
    /// the edge id; Simple/Acyclic: the node id); `None` pushes nothing (Walk).
    Go(Option<u32>),
}

/// The per-hop reuse gate shared by every var-length DFS. Decides whether the hop
/// across `a` is legal under `mode`, and whether it closes a Simple cycle.
///
/// For the node modes `used` is a NODE stack (the driver seeds it with `start`); for
/// Trail it is an EDGE stack. `Simple` permits a hop that closes the cycle on the
/// walk's `start` even though `start` is already marked — that hop emits (via
/// [`VarStep::Close`]) but terminates the path.
#[inline]
fn varlen_step(mode: PathMode, start: u32, a: &crate::store::Adj, used: &[u32]) -> VarStep {
    if matches!(mode, PathMode::Simple) && a.nbr == start {
        return VarStep::Close;
    }
    let collide = match mode {
        PathMode::Trail => used.contains(&a.eid),
        PathMode::Simple | PathMode::Acyclic => used.contains(&a.nbr),
        PathMode::Walk => false,
    };
    if collide {
        return VarStep::Skip;
    }
    let mark = match mode {
        PathMode::Walk => None,
        PathMode::Trail => Some(a.eid),
        PathMode::Simple | PathMode::Acyclic => Some(a.nbr),
    };
    VarStep::Go(mark)
}

/// Answer `count(DISTINCT endpoint)` over a bounded var-length hop by MULTI-SOURCE
/// BFS with a visited bitset — O(V+E) — instead of enumerating every path (with its
/// full multiplicity) and deduping the endpoints, which explodes with fan-out. The
/// DISTINCT endpoint set is exactly the nodes at shortest distance in `min..=max`
/// from the source set: a node with ANY walk of length `L ≤ max` has shortest
/// distance ≤ L, so a `min ≤ 1` reachability is the same set whether paths are
/// walks or trails (the shortest path is simple, reusing no edge). That equivalence
/// only holds for `min ≤ 1` (a node discovered at its shortest distance `< min`
/// might still be a valid longer-walk endpoint, which BFS would miss), so deeper
/// lower bounds fall back to the general path. The count is a set size, so it is
/// byte-identical to core's regardless of visitation order.
fn try_varlen_distinct_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    // count(DISTINCT <endpoint slot>) only.
    if agg.func != AggFn::Count || !agg.distinct {
        return None;
    }
    let Some(Expr::Slot(want_slot)) = agg.arg.as_ref() else {
        return None;
    };
    let Plan::VarLength {
        input: inner,
        from,
        dir,
        edge_label,
        min,
        max,
        mode,
        until,
        body_filter,
        double_loops: _, // a distinct endpoint set is blind to edge multiplicity
    } = input
    else {
        return None;
    };
    if until.is_some() || body_filter.is_some() {
        return None; // an until(pred) walk emits a filtered subset — no closed-form count
    }
    // A distinct ENDPOINT set is blind to edge multiplicity, so `double_loops` (a
    // both()-crossed self-loop counted twice) is irrelevant here — the self is reached
    // either way. It is deliberately NOT a bail condition (unlike the multiplicity
    // counts).
    //
    // The set-reachability fusion relies on nodes being allowed to repeat (Walk /
    // Trail). SIMPLE / ACYCLIC forbid node reuse, so a distinct-endpoint count must
    // enumerate — fall through.
    if !matches!(mode, PathMode::Walk | PathMode::Trail) {
        return None;
    }
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)), // unknown edge type → no endpoints
    };
    let batch = pull(inner, store, false).ok()?;
    // The endpoint the VarLength appends lands at the slot just past the inner width;
    // the DISTINCT arg must be exactly that endpoint (not, say, the source slot).
    if *want_slot != batch.slots.len() {
        return None;
    }
    let Col::Nodes(src) = batch.slot(*from) else {
        return None;
    };
    let count = varlen_distinct_endpoint_count(store, src, *dir, &want, *min, *max);
    Some(scalar_num(count as f64))
}

/// `<walk>.dedup().count()` — `count(*)` over a `DistinctBy` (on the endpoint slot) of a
/// var-length walk. Same distinct-endpoint count as [`try_varlen_distinct_count`]'s
/// `count(DISTINCT endpoint)`, but the front-end spells `dedup().count()` as a separate
/// `DistinctBy` node rather than a distinct aggregate, so it needs its own matcher. The
/// dedup key must be exactly the walk's appended endpoint slot; anything else (dedup on
/// the source, a multi-key dedup) is a different question and falls through.
fn try_varlen_distinctby_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None; // plain count(*)
    }
    let Plan::DistinctBy {
        input: inner,
        key_slots,
    } = input
    else {
        return None;
    };
    let [dedup_slot] = key_slots.as_slice() else {
        return None;
    };
    // The dedup'd source is either a var-length WALK (repeat(...).times(k)) or a single
    // Expand hop (`both().dedup()`) — both a distinct-endpoint question. Normalize to
    // (inner-plan, from, dir, edge_label, min, max); a single Expand is a 1-hop walk.
    let (src_plan, from, dir, edge_label, min, max) = match inner.as_ref() {
        Plan::VarLength {
            input: vl_inner,
            from,
            dir,
            edge_label,
            min,
            max,
            mode,
            until,
            body_filter,
            double_loops: _, // a distinct endpoint set is blind to edge multiplicity
        } => {
            if until.is_some()
                || body_filter.is_some()
                || !matches!(mode, PathMode::Walk | PathMode::Trail)
            {
                return None;
            }
            (vl_inner.as_ref(), *from, *dir, edge_label, *min, *max)
        }
        Plan::Expand {
            input: ex_inner,
            from,
            dir,
            edge_label,
            bind_edge,
            double_loops: _,
        } => {
            if *bind_edge {
                return None; // the bound edge slot shifts the endpoint; not this shape
            }
            // A single hop below the dedup; a deeper chain (`both().both().dedup()`) keeps
            // its already-good path — materialize the earlier hops, BFS the last — because
            // a full multi-level BFS from the scan re-expands the whole (dense) graph and
            // measured ~8x SLOWER on a dense 2-hop. (Rejected optimization, kept for the
            // note.)
            (ex_inner.as_ref(), *from, *dir, edge_label, 1u32, 1u32)
        }
        _ => return None,
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)),
    };
    let batch = pull(src_plan, store, false).ok()?;
    // The dedup key must be exactly the endpoint the hop appends (slot == inner width).
    if *dedup_slot != batch.slots.len() {
        return None;
    }
    let Col::Nodes(src) = batch.slot(from) else {
        return None;
    };
    let count = varlen_distinct_endpoint_count(store, src, dir, &want, min, max);
    Some(scalar_num(count as f64))
}

/// The number of DISTINCT nodes reachable from `src` by a Walk/Trail of `min..=max`
/// hops along `dir`/`want` edges — the shared kernel behind both `count(DISTINCT
/// endpoint)` and `count(*)` over a `dedup()` of a var-length walk. Two regimes:
///
/// - `min ≤ 1`: cumulative shortest-distance BFS. Each node is expanded at most once
///   (at its shortest distance), so an edge is traversed once — O(E) — and every node
///   within `max` hops is an endpoint (every hop ≥ 1 ≥ min).
/// - `min ≥ 2`: a walk may revisit, so the endpoints at EXACTLY h hops are the h-th
///   neighbour-set iterate N^h(src), NOT the distance-h set. Expand the DISTINCT
///   frontier one level at a time (each level a set — a node expands at most once per
///   level) and union the levels in `min..=max`. O(hops · E), versus the
///   product-of-degrees an enumeration of every walk would pay.
///
/// Blind to edge multiplicity (a set), so a both()-crossed self-loop needs no special
/// casing. `min == 0` also counts the sources as their own 0-hop endpoints.
fn varlen_distinct_endpoint_count(
    store: &Store,
    src: &[u32],
    dir: Dir,
    want: &[u32],
    min: u32,
    max: u32,
) -> usize {
    let n = store.node_count();
    if min >= 2 {
        let mut reached = vec![false; n];
        let mut in_next = vec![false; n];
        let mut seen = vec![false; n];
        let mut frontier: Vec<u32> = Vec::with_capacity(src.len());
        for &s in src {
            if !seen[s as usize] {
                seen[s as usize] = true;
                frontier.push(s);
            }
        }
        let mut next: Vec<u32> = Vec::new();
        for hop in 1..=max {
            if frontier.is_empty() {
                break;
            }
            next.clear();
            for &v in &frontier {
                for_each_nbr(store, v, dir, want, false, |nbr, _| {
                    if !in_next[nbr as usize] {
                        in_next[nbr as usize] = true;
                        next.push(nbr);
                    }
                });
            }
            if hop >= min {
                for &w in &next {
                    reached[w as usize] = true;
                }
            }
            // Reset the level-set for reuse (only the touched entries), then advance.
            for &w in &next {
                in_next[w as usize] = false;
            }
            std::mem::swap(&mut frontier, &mut next);
        }
        return reached.iter().filter(|&&r| r).count();
    }
    let mut visited = vec![false; n]; // added to a frontier (expansion dedup)
    let mut reached = vec![false; n]; // a valid endpoint (hop in min..=max)
    let mut frontier: Vec<u32> = Vec::with_capacity(src.len());
    for &s in src {
        if !visited[s as usize] {
            visited[s as usize] = true;
            frontier.push(s);
        }
        if min == 0 {
            reached[s as usize] = true; // the 0-hop path a=b
        }
    }
    let mut next: Vec<u32> = Vec::new();
    for _hop in 1..=max {
        if frontier.is_empty() {
            break;
        }
        for &v in &frontier {
            for_each_nbr(store, v, dir, want, false, |nbr, _| {
                reached[nbr as usize] = true;
                if !visited[nbr as usize] {
                    visited[nbr as usize] = true;
                    next.push(nbr);
                }
            });
        }
        std::mem::swap(&mut frontier, &mut next);
        next.clear();
    }
    reached.iter().filter(|&&r| r).count()
}

/// The fold twin of `varlen_count_dfs`: calls `emit(endpoint)` at every length in
/// `min..=max` instead of counting. Traversal / edge-type / trail logic — and thus
/// the EMISSION ORDER — are identical to `var_length`, so a `sum` folded here lands
/// the same value as materializing then summing.
#[allow(clippy::too_many_arguments)]
fn varlen_agg_dfs(
    store: &Store,
    v: u32,
    _len: u32, // always 0 at the call sites — the walk starts at the source
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    mode: PathMode,
    start: u32,
    used: &mut Vec<u32>,
    emit: &mut dyn FnMut(u32),
) {
    // The fold twin visits the same endpoints in the same order as the materializing
    // path (this fast-path is only taken without a `both()` double-loop, so `false`).
    varlen_scan_walk(
        store, v, min, max, dir, want, mode, start, used, false, emit,
    );
}

/// A scalar `sum`/`avg`/`min`/`max`/`count(arg)` over a bare var-length's ENDPOINT
/// property, folded DURING the DFS — no keep/ends, no gather, no intermediate batch
/// (which `try_frontier_aggregate`/`aggregate` all build, ~3x the traversal). The
/// emission order matches `var_length`, so `sum` folds in the same order and the
/// value contract (`cmp_num_total`) drives min/max — byte-identical to the
/// materializing path. `None` unless the aggregate reads exactly the appended
/// endpoint slot (block-streaming the general chain was measured a net regression;
/// this surgical fold is the low-overhead win).
fn try_varlen_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.distinct
        || !matches!(
            agg.func,
            AggFn::Sum | AggFn::Avg | AggFn::Min | AggFn::Max | AggFn::Count
        )
    {
        return None;
    }
    let Plan::VarLength {
        input: inner,
        from,
        dir,
        edge_label,
        min,
        max,
        mode,
        until,
        body_filter,
        double_loops,
    } = input
    else {
        return None;
    };
    if until.is_some() || body_filter.is_some() || *double_loops {
        return None; // an until(pred) walk emits a filtered subset — no closed-form agg
    }
    // The aggregate argument must be a property of the ENDPOINT (the appended slot).
    let Some(Expr::Prop { slot, key }) = agg.arg.as_ref() else {
        return None; // count(*) is `try_varlen_count`
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        // Unknown edge type → no paths. A non-empty want of a non-existent id
        // (etype ids are dense, so u32::MAX is none) matches nothing, yielding the
        // empty-aggregate value without a special-cased early return here.
        Err(()) => vec![u32::MAX],
    };
    let batch = pull(inner, store, false).ok()?;
    if *slot != batch.slots.len() {
        return None; // arg is not the endpoint
    }
    let Col::Nodes(src) = batch.slot(*from) else {
        return None;
    };
    let column = store.column(key)?; // property absent everywhere → fall back
    let node_unique = matches!(mode, PathMode::Simple | PathMode::Acyclic);
    let dfs = |emit: &mut dyn FnMut(u32)| {
        let mut used: Vec<u32> = Vec::new();
        for &v in src {
            if node_unique {
                used.push(v); // mark the start node
            }
            varlen_agg_dfs(
                store, v, 0, *min, *max, *dir, &want, *mode, v, &mut used, emit,
            );
            if node_unique {
                used.pop();
            }
        }
    };

    let val = match (agg.func, column) {
        (AggFn::Sum | AggFn::Avg, Column::Num { data, present, .. }) => {
            let mut total = 0.0f64;
            let mut cnt = 0u64;
            dfs(&mut |v| {
                let i = v as usize;
                if present[i] {
                    total += data[i];
                    cnt += 1;
                }
            });
            if agg.func == AggFn::Sum {
                Value::Num(total)
            } else if cnt == 0 {
                Value::Null
            } else {
                Value::Num(total / cnt as f64)
            }
        }
        (AggFn::Min | AggFn::Max, Column::Num { data, present, .. }) => {
            let want_min = agg.func == AggFn::Min;
            let mut best: Option<f64> = None;
            dfs(&mut |v| {
                let i = v as usize;
                if present[i] {
                    let x = data[i];
                    best = Some(match best {
                        None => x,
                        Some(b) => {
                            let ord = value::cmp_num_total(x, b);
                            if (want_min && ord.is_lt()) || (!want_min && ord.is_gt()) {
                                x
                            } else {
                                b
                            }
                        }
                    });
                }
            });
            best.map_or(Value::Null, Value::Num)
        }
        (AggFn::Min | AggFn::Max, Column::Str { data, present, .. }) => {
            // Track the best endpoint id (not a borrow into `data`), comparing `&str`
            // directly — the value contract's order for two strings is lexicographic,
            // so this equals the materializing min/max. `<`/`>` on equal keeps the
            // first (`cmp_total(..).is_lt()` semantics).
            let want_min = agg.func == AggFn::Min;
            let mut best: Option<u32> = None;
            dfs(&mut |v| {
                let i = v as usize;
                if present[i] {
                    best = Some(match best {
                        None => v,
                        Some(b) => {
                            let (sv, sb) = (data[i].as_ref(), data[b as usize].as_ref());
                            if (want_min && sv < sb) || (!want_min && sv > sb) {
                                v
                            } else {
                                b
                            }
                        }
                    });
                }
            });
            best.map_or(Value::Null, |v| Value::Str(data[v as usize].clone()))
        }
        (
            AggFn::Min | AggFn::Max,
            Column::Dict {
                dict,
                codes,
                present,
                ..
            },
        ) => {
            let want_min = agg.func == AggFn::Min;
            let str_of = |v: u32| dict[codes[v as usize] as usize].as_ref();
            let mut best: Option<u32> = None;
            dfs(&mut |v| {
                if present[v as usize] {
                    best = Some(match best {
                        None => v,
                        Some(b) => {
                            if (want_min && str_of(v) < str_of(b))
                                || (!want_min && str_of(v) > str_of(b))
                            {
                                v
                            } else {
                                b
                            }
                        }
                    });
                }
            });
            best.map_or(Value::Null, |v| {
                Value::Str(dict[codes[v as usize] as usize].clone())
            })
        }
        (AggFn::Min | AggFn::Max, _) => return None, // Temporal/Bool/Gen → general path
        (AggFn::Count, col) => {
            // count(arg): endpoints whose property is present (non-null).
            let present: &[bool] = match col {
                Column::Num { present, .. }
                | Column::Str { present, .. }
                | Column::Bool { present, .. }
                | Column::Dict { present, .. } => present,
                _ => return None, // Temporal/Gen → the general path
            };
            let mut cnt = 0u64;
            dfs(&mut |v| {
                if present[v as usize] {
                    cnt += 1;
                }
            });
            Value::Num(cnt as f64)
        }
        _ => return None,
    };
    Some(Batch::single(Col::Gen(vec![val])))
}

/// `<hop-chain>.values(k).min()/max()` over a pure Scan/Expand chain: fold the numeric
/// property `k` over the per-node PATH-COUNT frontier — WITHOUT materializing the
/// exploding frontier (the min/max analog of the count fast-path). MIN/MAX only: they
/// are order-INDEPENDENT, so collapsing the frontier to per-node multiplicity (which
/// loses row order) is byte-identical; SUM/AVG would change the summation order, so they
/// stay on `try_frontier_aggregate` (`frontier_ids`, which keeps order). Numeric columns
/// only; a filtered chain / edge hop / non-numeric column returns None.
/// `count(*)` over any plan `frontier_counts` can fold — including a `hasLabel(L)`-
/// filtered hop chain (`<hops>.hasLabel(L).count()`) — as the SUM of the per-node path
/// multiplicities, never materializing the frontier. Order-independent (an integer row
/// count), so byte-identical. `None` for a non-fusable shape.
fn try_frontier_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None; // count(*) only
    }
    // Only worth the frontier fold when a plain filter/hop chain would otherwise
    // materialize — i.e. there IS a frontier filter (the bare-chain counts already have
    // their own fast-paths). Require a top-level IsLabeled filter.
    if !matches!(
        input,
        Plan::Filter {
            pred: Expr::IsLabeled { .. },
            ..
        }
    ) {
        return None;
    }
    let counts = frontier_counts(input, store)?;
    let mut total = 0f64;
    counts.for_each(|_, c| total += c);
    Some(scalar_num(total))
}

fn try_frontier_prop_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.distinct || !matches!(agg.func, AggFn::Min | AggFn::Max) {
        return None;
    }
    let Some(Expr::Prop { slot, key }) = agg.arg.as_ref() else {
        return None;
    };
    let width = chain_width(input)?;
    if *slot != width - 1 {
        return None; // arg must be a property of the chain frontier
    }
    let Some(Column::Num { data, present, .. }) = store.column(key) else {
        return None; // non-numeric / absent-everywhere → general path
    };
    let counts = frontier_counts(input, store)?;
    let want_min = agg.func == AggFn::Min;
    let mut best: Option<f64> = None;
    counts.for_each(|v, _c| {
        let i = v as usize;
        if present[i] {
            let x = data[i];
            best = Some(match best {
                None => x,
                Some(b) => {
                    let ord = value::cmp_num_total(x, b);
                    if (want_min && ord.is_lt()) || (!want_min && ord.is_gt()) {
                        x
                    } else {
                        b
                    }
                }
            });
        }
    });
    Some(Batch::single(Col::Gen(vec![
        best.map_or(Value::Null, Value::Num)
    ])))
}

/// Answer a scalar `count(*)` over a bare labelled/unlabelled `Scan` in O(1) (a
/// label bucket length — buckets hold only live ids) or a single tombstone-bitmap
/// sweep (unlabelled), WITHOUT materializing the id vector. `None` for any other
/// shape (a WHERE seed, an Expand, `count(arg)`), which the other paths handle.
fn try_scan_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() {
        return None; // count(*) only; count(arg)/DISTINCT need the values
    }
    let n = match input {
        Plan::Scan { label: Some(l) } => store.nodes_with_label(l).len(),
        Plan::Scan { label: None } => store.live_node_count(),
        _ => return None,
    };
    Some(scalar_num(n as f64))
}

/// Answer a scalar `sum`/`avg`/`count(arg)` over a bare `Scan`'s Num property by
/// summing the RAW f64 column (present cells only), WITHOUT materializing the
/// frontier or boxing each cell into a `Value`. `None` (fall back) for a grouped
/// aggregate, a DISTINCT, `min`/`max` (need the value-contract order), a non-`Num`
/// column (which may need poison handling), or any non-`Scan` input.
/// A scalar numeric aggregate (`sum`/`avg`/`min`/`max`/`count(prop)`) over a FILTERED
/// scan — `has(...).values(k).sum()` and friends. Get the survivors from the filter fast
/// path (`try_filter_keep`, which raw-passes num/str/And/Not/dict predicates), then
/// accumulate the aggregate directly over their column values — no gather of a survivor
/// frontier, no Project into a Col::Num, no boxed fold. `None` for a non-`Filter{Scan}`
/// input, a non-fast-pathable predicate, or an unsupported agg — the general path runs.
fn try_filtered_scan_num_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.distinct
        || !matches!(
            agg.func,
            AggFn::Sum | AggFn::Avg | AggFn::Min | AggFn::Max | AggFn::Count
        )
    {
        return None;
    }
    let Plan::Filter { input: scan, pred } = input else {
        return None;
    };
    let Plan::Scan { label } = scan.as_ref() else {
        return None;
    };
    let Some(Expr::Prop { slot: 0, key }) = agg.arg.as_ref() else {
        return None; // count(*) is try_filtered_count; only a prop agg here
    };
    let Some(Column::Num { data, present, .. }) = store.column(key) else {
        return None;
    };
    // Survivor ROWS of the scan frontier (row index == the node id for a bare scan).
    let ids: Vec<u32> = match label {
        Some(l) => store.nodes_with_label(l).to_vec(),
        None => store.all_nodes(),
    };
    let batch = Batch::of(vec![Col::Nodes(ids)]);
    let keep = try_filter_keep(pred, store, &batch)?;
    let Col::Nodes(sids) = batch.slot(0) else {
        return None;
    };
    let (mut total, mut cnt, mut best): (f64, u64, Option<f64>) = (0.0, 0, None);
    for &row in &keep {
        let i = sids[row] as usize;
        if !present[i] {
            continue; // the agg's own prop may be NULL even when the filter passed
        }
        let x = data[i];
        total += x;
        cnt += 1;
        best = Some(match best {
            None => x,
            Some(b) => {
                let keep_new = (agg.func == AggFn::Min && value::cmp_num_total(x, b).is_lt())
                    || (agg.func == AggFn::Max && value::cmp_num_total(x, b).is_gt());
                if keep_new {
                    x
                } else {
                    b
                }
            }
        });
    }
    let result = match agg.func {
        AggFn::Sum if agg.null_on_empty && cnt == 0 => Value::Null,
        AggFn::Sum => Value::Num(total),
        AggFn::Count => Value::Num(cnt as f64),
        AggFn::Avg => {
            if cnt == 0 {
                Value::Null
            } else {
                Value::Num(total / cnt as f64)
            }
        }
        _ => best.map_or(Value::Null, Value::Num), // min/max of nothing → NULL
    };
    Some(Batch::single(Col::Gen(vec![result])))
}

fn try_scan_num_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.distinct || !matches!(agg.func, AggFn::Sum | AggFn::Avg | AggFn::Count) {
        return None;
    }
    let label = match input {
        Plan::Scan { label } => label,
        _ => return None,
    };
    let Some(Expr::Prop { slot: 0, key }) = agg.arg.as_ref() else {
        return None;
    };
    let Some(Column::Num { data, present, .. }) = store.column(key) else {
        return None; // non-numeric column: the general path handles poison
    };
    let (mut total, mut cnt) = (0f64, 0u64);
    // Whole-column fast path: when the scan covers EVERY live node (an unlabelled
    // scan, or a label all nodes carry) with nothing deleted, sum the raw
    // `data`/`present` slices directly — no per-row id indirection, so the loop
    // auto-vectorizes. Otherwise walk the label's id list.
    let all_live = store.live_node_count() == store.node_count();
    let whole = all_live
        && match label {
            None => true,
            Some(l) => store.nodes_with_label(l).len() == store.node_count(),
        };
    if whole {
        for (i, &x) in data.iter().enumerate() {
            if present[i] {
                total += x;
                cnt += 1;
            }
        }
    } else {
        let mut visit = |i: usize| {
            if present[i] {
                total += data[i];
                cnt += 1;
            }
        };
        match label {
            Some(l) => store
                .nodes_with_label(l)
                .iter()
                .for_each(|&id| visit(id as usize)),
            None => (0..store.node_count()).for_each(|i| {
                if store.is_alive(i as u32) {
                    visit(i);
                }
            }),
        }
    }
    let result = match agg.func {
        AggFn::Sum if agg.null_on_empty && cnt == 0 => Value::Null, // Gremlin sum() of nothing
        AggFn::Sum => Value::Num(total), // 0.0 over an empty/all-null set (K0a)
        AggFn::Count => Value::Num(cnt as f64), // count(arg) = present count
        _ => {
            if cnt == 0 {
                Value::Null // avg of nothing
            } else {
                Value::Num(total / cnt as f64)
            }
        }
    };
    Some(Batch::of(vec![Col::Gen(vec![result])]))
}

/// Visit each scanned node's dense id (as `usize`) for a bare `Scan`. Iterates the
/// raw `0..node_count` range directly when the scan covers every live node (an
/// unlabelled scan, or a label all nodes carry, nothing deleted) — sequential and
/// vectorizable — otherwise walks the label's id list. Generic over `F` so there is
/// no per-node dynamic dispatch. Shared by the scan-aggregate fast paths.
fn scan_visit<F: FnMut(usize)>(store: &Store, label: &Option<String>, mut f: F) {
    let all_live = store.live_node_count() == store.node_count();
    let whole = all_live
        && match label {
            None => true,
            Some(l) => store.nodes_with_label(l).len() == store.node_count(),
        };
    if whole {
        (0..store.node_count()).for_each(&mut f);
    } else {
        match label {
            Some(l) => store
                .nodes_with_label(l)
                .iter()
                .for_each(|&id| f(id as usize)),
            None => (0..store.node_count()).for_each(|i| {
                if store.is_alive(i as u32) {
                    f(i);
                }
            }),
        }
    }
}

/// A group's accumulators: row count (for `count(*)`) plus `(total, count, best)`
/// per numeric aggregate.
struct GroupAcc {
    rows: u64,
    aggs: Vec<(f64, u64, Option<f64>)>,
}

/// Fused single-key grouped aggregate over a bare `Scan`: `RETURN n.k AS key,
/// <aggs> …` where the group key is a `Str`/`Num`/`Bool` column and each aggregate
/// is `count(*)` or a numeric reduction over a `Num` column. Reads the storage
/// columns directly and groups by the TYPED key value (first-seen order, matching
/// the grouping contract), so the frontier and projected columns are never
/// materialized. `None` for any other shape (Temporal/Gen key, non-numeric agg
/// arg, DISTINCT, multi-key). The per-key string hashing is the residual floor.
/// The TIGHT case of [`try_scan_group_agg`]: a plain `count(*) GROUP BY <col>` where the
/// group column is DICTIONARY-encoded (a categorical `city`/`status`). Count directly per
/// dict CODE into a `Vec<u64>` — no per-group `GroupAcc` struct, no `accumulate` closure,
/// no bounds-checked `acc[group]` write per row, just `counts[code] += 1`.
///
/// This exists because the general `GroupAcc` path, while fine natively, is
/// DISPROPORTIONATELY slow on wasm: its nested closure + per-row struct indexing compile
/// to indirect calls / bounds-checked accesses that wasm penalizes several times more
/// than native, which flipped `groupCount().by('city')` and `GROUP BY city` from wins to
/// losses on the wasm surface while they won on FFI/native. The lean loop closes that.
/// Numeric aggregates, multi-agg, and non-dict keys stay on `try_scan_group_agg`.
fn try_scan_dict_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    let [(_, Expr::Prop { slot: 0, key })] = keys else {
        return None;
    };
    let [agg] = aggs else {
        return None;
    };
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None; // count(*) only
    }
    let Plan::Scan { label } = input else {
        return None;
    };
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(key)
    else {
        return None;
    };
    // `usize` counters, NOT u64: a count cannot exceed node_count, and `usize` is 32-bit
    // (native i32) on wasm32 where a u64 add is EMULATED — the general path's u64
    // GroupAcc.rows is part of why grouping was disproportionately slow on wasm.
    let mut counts = vec![0usize; dict.len()];
    let mut null_count = 0usize;
    // Group output order is FIRST-SEEN (the grouping contract, matching the general path):
    // record a code the first time it is counted (its count goes 0 -> 1); -1 = null group.
    let mut order: Vec<i32> = Vec::new();
    let mut seen_null = false;
    scan_visit(store, label, |i| {
        if present[i] {
            let c = codes[i] as usize;
            if counts[c] == 0 {
                order.push(c as i32);
            }
            counts[c] += 1;
        } else {
            if !seen_null {
                seen_null = true;
                order.push(-1);
            }
            null_count += 1;
        }
    });
    let mut key_col: Vec<Value> = Vec::with_capacity(order.len());
    let mut cnt_col: Vec<Value> = Vec::with_capacity(order.len());
    for &code in &order {
        if code < 0 {
            key_col.push(Value::Null);
            cnt_col.push(Value::Num(null_count as f64));
        } else {
            key_col.push(Value::Str(dict[code as usize].clone()));
            cnt_col.push(Value::Num(counts[code as usize] as f64));
        }
    }
    Some(Batch::of(vec![Col::Gen(key_col), Col::Gen(cnt_col)]))
}

/// `count(*) GROUP BY <dict col>` over ANY frontier — a hop, a filter — not just a bare
/// Scan (which [`try_scan_dict_count`] already streams). Pull the frontier once, then
/// count per dict CODE, instead of the general group_by decoding each cell to a
/// `Value::Str` and HASHING it (`group_by_arc`) — the string hash over a big hop frontier
/// was the whole cost (`inE('R').outV().groupCount().by('city')` at 0.08x). First-seen
/// group order and null handling (absent OR the u32::MAX optional-match sentinel → the
/// null group) match the general path exactly.
fn try_frontier_dict_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
    track: bool,
) -> Option<Batch> {
    let [(_, Expr::Prop { slot, key })] = keys else {
        return None;
    };
    let [agg] = aggs else {
        return None;
    };
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None;
    }
    if matches!(input, Plan::Scan { .. }) {
        return None; // the bare-scan case streams via try_scan_dict_count
    }
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(key)
    else {
        return None;
    };
    let batch = pull(input, store, track).ok()?;
    let Col::Nodes(ids) = batch.slot(*slot) else {
        return None;
    };
    let mut counts = vec![0usize; dict.len()];
    let mut null_count = 0usize;
    let mut order: Vec<i32> = Vec::new();
    let mut seen_null = false;
    for &id in ids {
        if id != u32::MAX && present[id as usize] {
            let c = codes[id as usize] as usize;
            if counts[c] == 0 {
                order.push(c as i32);
            }
            counts[c] += 1;
        } else {
            if !seen_null {
                seen_null = true;
                order.push(-1);
            }
            null_count += 1;
        }
    }
    let mut key_col: Vec<Value> = Vec::with_capacity(order.len());
    let mut cnt_col: Vec<Value> = Vec::with_capacity(order.len());
    for &code in &order {
        if code < 0 {
            key_col.push(Value::Null);
            cnt_col.push(Value::Num(null_count as f64));
        } else {
            key_col.push(Value::Str(dict[code as usize].clone()));
            cnt_col.push(Value::Num(counts[code as usize] as f64));
        }
    }
    Some(Batch::of(vec![Col::Gen(key_col), Col::Gen(cnt_col)]))
}

fn try_scan_group_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    let [(_, Expr::Prop { slot: 0, key: gkey })] = keys else {
        return None;
    };
    let label = match input {
        Plan::Scan { label } => label,
        _ => return None,
    };
    // Agg specs: the Num column (None for count(*)) and function.
    type Spec<'a> = (Option<(&'a [f64], &'a [bool])>, AggFn);
    let mut specs: Vec<Spec> = Vec::with_capacity(aggs.len());
    for agg in aggs {
        if agg.distinct {
            return None;
        }
        match (agg.func, agg.arg.as_ref()) {
            (AggFn::Count, None) => specs.push((None, AggFn::Count)),
            (
                AggFn::Sum | AggFn::Avg | AggFn::Count | AggFn::Min | AggFn::Max,
                Some(Expr::Prop { slot: 0, key }),
            ) => {
                let Some(Column::Num { data, present, .. }) = store.column(key) else {
                    return None;
                };
                specs.push((Some((data.as_slice(), present.as_slice())), agg.func));
            }
            _ => return None,
        }
    }

    let mut group_keys: Vec<Value> = Vec::new();
    let mut acc: Vec<GroupAcc> = Vec::new();
    let na = specs.len();
    // Add one row (dense group id `g`) to the accumulators.
    let accumulate = |acc: &mut Vec<GroupAcc>, g: usize, i: usize| {
        let a = &mut acc[g];
        a.rows += 1;
        for (k, &(col, func)) in specs.iter().enumerate() {
            let Some((data, present)) = col else { continue };
            if !present[i] {
                continue;
            }
            let x = data[i];
            let s = &mut a.aggs[k];
            s.0 += x;
            s.1 += 1;
            s.2 = Some(match s.2 {
                None => x,
                Some(b) => match func {
                    AggFn::Min if value::cmp_num_total(x, b).is_lt() => x,
                    AggFn::Max if value::cmp_num_total(x, b).is_gt() => x,
                    _ => b,
                },
            });
        }
    };

    // Resolve a row to a dense group id (first-seen), creating the group on demand.
    macro_rules! run {
        ($present:expr, $lookup:expr, $keyval:expr, $nullkey:expr) => {{
            let present = $present;
            let mut map: FnvMap<_, u32> = FnvMap::default();
            let mut null_group: Option<u32> = None;
            scan_visit(store, label, |i| {
                let g = if present[i] {
                    let k = $lookup(i);
                    match map.get(&k) {
                        Some(&g) => g as usize,
                        None => {
                            let g = group_keys.len() as u32;
                            map.insert(k, g);
                            group_keys.push($keyval(i));
                            acc.push(GroupAcc {
                                rows: 0,
                                aggs: vec![(0.0, 0, None); na],
                            });
                            g as usize
                        }
                    }
                } else {
                    match null_group {
                        Some(g) => g as usize,
                        None => {
                            let g = group_keys.len() as u32;
                            null_group = Some(g);
                            group_keys.push(Value::Null);
                            acc.push(GroupAcc {
                                rows: 0,
                                aggs: vec![(0.0, 0, None); na],
                            });
                            g as usize
                        }
                    }
                };
                accumulate(&mut acc, g, i);
            });
            let _ = $nullkey; // silence unused when the key type has no null path
        }};
    }
    // Only a STRING group key: reading the storage column directly avoids
    // materializing 100k `Arc<str>` (the win). A Num/Bool key already groups via
    // `assign_groups`' typed fast path over the materialized column, which is as
    // fast — so leave those to the general aggregate (this fused path's per-agg
    // accumulator loop is slightly heavier and would regress them).
    match store.column(gkey)? {
        Column::Str { data, present, .. } => {
            run!(
                present,
                |i: usize| data[i].as_ref(),
                |i: usize| Value::Str(data[i].clone()),
                ()
            );
        }
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            // Group by CODE, mapped to a dense group id in first-seen (scan) order —
            // a per-code slot, no per-row string hash. First-seen (not dict order) is
            // what the pinned GROUP BY order requires, since the dict was built over
            // all nodes and the scan may visit a label subset in a different order.
            let mut code_to_group: Vec<u32> = vec![u32::MAX; dict.len()];
            let mut null_group: Option<u32> = None;
            scan_visit(store, label, |i| {
                let g = if present[i] {
                    let c = codes[i] as usize;
                    if code_to_group[c] == u32::MAX {
                        let g = group_keys.len() as u32;
                        code_to_group[c] = g;
                        group_keys.push(Value::Str(dict[c].clone()));
                        acc.push(GroupAcc {
                            rows: 0,
                            aggs: vec![(0.0, 0, None); na],
                        });
                        g as usize
                    } else {
                        code_to_group[c] as usize
                    }
                } else {
                    match null_group {
                        Some(g) => g as usize,
                        None => {
                            let g = group_keys.len() as u32;
                            null_group = Some(g);
                            group_keys.push(Value::Null);
                            acc.push(GroupAcc {
                                rows: 0,
                                aggs: vec![(0.0, 0, None); na],
                            });
                            g as usize
                        }
                    }
                };
                accumulate(&mut acc, g, i);
            });
        }
        _ => return None,
    }

    // Build the output: the key column, then one column per aggregate.
    let key_col = Col::Gen(group_keys);
    let mut cols = vec![key_col];
    for (k, &(col, func)) in specs.iter().enumerate() {
        let vals: Vec<Value> = acc
            .iter()
            .map(|a| {
                let (total, cnt, best) = a.aggs[k];
                match func {
                    AggFn::Count if col.is_none() => Value::Num(a.rows as f64),
                    AggFn::Count => Value::Num(cnt as f64),
                    AggFn::Sum => Value::Num(total),
                    AggFn::Avg => {
                        if cnt == 0 {
                            Value::Null
                        } else {
                            Value::Num(total / cnt as f64)
                        }
                    }
                    _ => best.map_or(Value::Null, Value::Num),
                }
            })
            .collect();
        cols.push(Col::Gen(vals));
    }
    Some(Batch::of(cols))
}

/// Answer `count(DISTINCT n.k)` over a bare `Scan` by deduping the RAW column into
/// a typed set (a `&str`, the f64 group bits, or a bool) and returning its size —
/// no frontier materialization and no per-cell byte-key serialization. Nulls are
/// skipped (as `count(DISTINCT)` does). `None` for a non-`Scan` input, a
/// Temporal/Gen column, or a non-distinct/`count(*)` agg.
/// A membership bitset over the DISTINCT integer values of a Num column: returns
/// `(min, bits)` where `bits[k]` is set iff the value `min + k` is present. Used
/// instead of hashing when every present value is a finite INTEGER in a small span
/// — `count(DISTINCT age)` / `DISTINCT age` over 100 ages then sets 100 bits rather
/// than hashing 200k cells. One pass finds the span + integrality (a non-integer,
/// NaN, or Inf value disqualifies via `fract()`/`is_finite`), a second sets the
/// bits. Distinct finite integers map to distinct offsets, so a popcount equals the
/// FnvSet's `len` and the set bits recover every distinct value exactly. `None`
/// (fall back to hashing) when the column is empty, non-integer, or spans too wide.
fn low_card_int_bitset(
    store: &Store,
    label: &Option<String>,
    data: &[f64],
    present: &[bool],
) -> Option<(f64, Vec<bool>, bool)> {
    const MAX_SPAN: usize = 1 << 20; // cap the bitset at ~1M bits (128 KB)
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut any, mut all_int, mut saw_absent) = (false, true, false);
    scan_visit(store, label, |i| {
        if present[i] {
            let x = data[i];
            any = true;
            if x.is_finite() && x.fract() == 0.0 {
                lo = lo.min(x);
                hi = hi.max(x);
            } else {
                all_int = false;
            }
        } else {
            saw_absent = true; // a NULL cell — DISTINCT keeps one, count ignores it
        }
    });
    if !any || !all_int {
        return None;
    }
    let span = (hi - lo) as usize;
    if span >= MAX_SPAN {
        return None;
    }
    let mut bits = vec![false; span + 1];
    scan_visit(store, label, |i| {
        if present[i] {
            bits[(data[i] - lo) as usize] = true;
        }
    });
    Some((lo, bits, saw_absent))
}

fn try_scan_distinct_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || !agg.distinct {
        return None;
    }
    let label = match input {
        Plan::Scan { label } => label,
        _ => return None,
    };
    let Some(Expr::Prop { slot: 0, key }) = agg.arg.as_ref() else {
        return None;
    };
    let count = match store.column(key)? {
        Column::Str { data, present, .. } => {
            let mut seen: FnvSet<&str> = FnvSet::default();
            scan_visit(store, label, |i| {
                if present[i] {
                    seen.insert(data[i].as_ref());
                }
            });
            seen.len()
        }
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            // A distinct value == a distinct code: mark a per-code bitset, no hashing.
            let mut seen = vec![false; dict.len()];
            scan_visit(store, label, |i| {
                if present[i] {
                    seen[codes[i] as usize] = true;
                }
            });
            seen.iter().filter(|&&b| b).count()
        }
        Column::Num { data, present, .. } => {
            // Low-cardinality integer fast path: dedup with a bitset (popcount), no
            // hashing. Falls back to the FnvSet when values are wide-ranged or
            // non-integer. The distinct count is identical either way.
            if let Some((_, bits, _)) = low_card_int_bitset(store, label, data, present) {
                bits.iter().filter(|&&b| b).count()
            } else {
                let mut seen: FnvSet<u64> = FnvSet::default();
                scan_visit(store, label, |i| {
                    if present[i] {
                        seen.insert(value::num_group_bits(data[i]));
                    }
                });
                seen.len()
            }
        }
        Column::Bool { data, present, .. } => {
            let mut seen = [false; 2];
            scan_visit(store, label, |i| {
                if present[i] {
                    seen[usize::from(data[i])] = true;
                }
            });
            usize::from(seen[0]) + usize::from(seen[1])
        }
        _ => return None, // Temporal / Gen → the general aggregate
    };
    Some(scalar_num(count as f64))
}

/// The frontier sibling of [`try_scan_distinct_count`]: `count(DISTINCT <frontier prop>)`
/// over a hop chain, deduped over the DISTINCT reached endpoints (`frontier_counts`) rather
/// than materializing the exploded path multiset and byte-keying every row through the
/// general grouped fold. Path multiplicity is irrelevant to DISTINCT, so visiting each
/// endpoint once yields the identical value set — byte-identical count, far less work.
fn try_frontier_distinct_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || !agg.distinct {
        return None;
    }
    let Some(Expr::Prop { slot, key }) = agg.arg.as_ref() else {
        return None;
    };
    let width = chain_width(input)?;
    if *slot != width - 1 {
        return None; // arg must be a property of the chain frontier
    }
    let counts = frontier_counts(input, store)?;
    let count = match store.column(key)? {
        Column::Str { data, present, .. } => {
            let mut seen: FnvSet<&str> = FnvSet::default();
            counts.for_each(|v, _| {
                let i = v as usize;
                if present[i] {
                    seen.insert(data[i].as_ref());
                }
            });
            seen.len()
        }
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            let mut seen = vec![false; dict.len()];
            counts.for_each(|v, _| {
                let i = v as usize;
                if present[i] {
                    seen[codes[i] as usize] = true;
                }
            });
            seen.iter().filter(|&&b| b).count()
        }
        Column::Num { data, present, .. } => {
            let mut seen: FnvSet<u64> = FnvSet::default();
            counts.for_each(|v, _| {
                let i = v as usize;
                if present[i] {
                    seen.insert(value::num_group_bits(data[i]));
                }
            });
            seen.len()
        }
        Column::Bool { data, present, .. } => {
            let mut seen = [false; 2];
            counts.for_each(|v, _| {
                let i = v as usize;
                if present[i] {
                    seen[usize::from(data[i])] = true;
                }
            });
            usize::from(seen[0]) + usize::from(seen[1])
        }
        _ => return None, // Temporal / Gen → the general aggregate
    };
    Some(scalar_num(count as f64))
}

/// Answer several scalar numeric aggregates (`sum`/`avg`/`min`/`max`/`count`) over
/// a bare `Scan` in ONE pass over the Num columns — e.g. `min(age), max(age)` or
/// `count(*), avg(age)`. `None` if any agg is grouped/DISTINCT or not a numeric
/// reduction over a `Num` property (or `count(*)`). Complements the single-agg
/// [`try_scan_num_agg`], which keeps the tighter auto-vectorized loop.
fn try_scan_multi_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.is_empty() {
        return None;
    }
    let label = match input {
        Plan::Scan { label } => label,
        _ => return None,
    };
    // Per agg: its Num column slices (None for `count(*)`) and function.
    type AggSpec<'a> = (Option<(&'a [f64], &'a [bool])>, AggFn);
    let mut specs: Vec<AggSpec> = Vec::with_capacity(aggs.len());
    for agg in aggs {
        if agg.distinct {
            return None;
        }
        match (agg.func, agg.arg.as_ref()) {
            (AggFn::Count, None) => specs.push((None, AggFn::Count)), // count(*)
            (
                AggFn::Sum | AggFn::Avg | AggFn::Count | AggFn::Min | AggFn::Max,
                Some(Expr::Prop { slot: 0, key }),
            ) => {
                let Some(Column::Num { data, present, .. }) = store.column(key) else {
                    return None;
                };
                specs.push((Some((data.as_slice(), present.as_slice())), agg.func));
            }
            _ => return None,
        }
    }
    // Fast path: every value-aggregate reads ONE Num column (e.g. `sum(age),
    // min(age), max(age)`) — a single BRANCH-FREE pass computing sum/cnt/min/max
    // with straight f64 ops (stored Nums are finite, so `x < mn` == cmp_num_total),
    // instead of the per-element per-spec match in the general loop below.
    let used: Vec<*const f64> = specs
        .iter()
        .filter_map(|(c, _)| c.map(|(d, _)| d.as_ptr()))
        .collect();
    if !used.is_empty() && used.iter().all(|&p| p == used[0]) {
        let (data, present) = specs.iter().find_map(|(c, _)| *c).expect("used non-empty");
        let (mut sum, mut cnt, mut mn, mut mx, mut rows) =
            (0.0f64, 0u64, f64::INFINITY, f64::NEG_INFINITY, 0u64);
        scan_visit(store, label, |i| {
            rows += 1;
            if present[i] {
                let x = data[i];
                sum += x;
                cnt += 1;
                if x < mn {
                    mn = x;
                }
                if x > mx {
                    mx = x;
                }
            }
        });
        let cols: Vec<Col> = specs
            .iter()
            .map(|&(col, func)| {
                let v = match func {
                    AggFn::Count if col.is_none() => Value::Num(rows as f64),
                    AggFn::Count => Value::Num(cnt as f64),
                    AggFn::Sum => Value::Num(sum),
                    AggFn::Avg if cnt == 0 => Value::Null,
                    AggFn::Avg => Value::Num(sum / cnt as f64),
                    AggFn::Min if cnt == 0 => Value::Null,
                    AggFn::Min => Value::Num(mn),
                    AggFn::Max if cnt == 0 => Value::Null,
                    _ => Value::Num(mx),
                };
                Col::Gen(vec![v])
            })
            .collect();
        return Some(Batch::of(cols));
    }
    // (total, count, best) per agg; `rows` counts scanned nodes for count(*).
    let mut acc: Vec<(f64, u64, Option<f64>)> = vec![(0.0, 0, None); specs.len()];
    let mut rows = 0u64;
    let mut visit = |i: usize| {
        rows += 1;
        for (k, (col, func)) in specs.iter().enumerate() {
            let Some((data, present)) = col else { continue };
            if !present[i] {
                continue;
            }
            let x = data[i];
            let a = &mut acc[k];
            a.0 += x;
            a.1 += 1;
            a.2 = Some(match a.2 {
                None => x,
                Some(b) => match func {
                    AggFn::Min if value::cmp_num_total(x, b).is_lt() => x,
                    AggFn::Max if value::cmp_num_total(x, b).is_gt() => x,
                    _ => b,
                },
            });
        }
    };
    let all_live = store.live_node_count() == store.node_count();
    let whole = all_live
        && match label {
            None => true,
            Some(l) => store.nodes_with_label(l).len() == store.node_count(),
        };
    if whole {
        (0..store.node_count()).for_each(&mut visit);
    } else {
        match label {
            Some(l) => store
                .nodes_with_label(l)
                .iter()
                .for_each(|&id| visit(id as usize)),
            None => (0..store.node_count()).for_each(|i| {
                if store.is_alive(i as u32) {
                    visit(i);
                }
            }),
        }
    }
    // One output COLUMN per aggregate, each a single row (a scalar aggregate emits
    // exactly one row).
    let cols: Vec<Col> = specs
        .iter()
        .zip(&acc)
        .map(|(&(col, func), &(total, cnt, best))| {
            let v = match func {
                AggFn::Count if col.is_none() => Value::Num(rows as f64), // count(*)
                AggFn::Count => Value::Num(cnt as f64),                   // count(arg)
                AggFn::Sum => Value::Num(total),                          // 0.0 over empty (K0a)
                AggFn::Avg => {
                    if cnt == 0 {
                        Value::Null
                    } else {
                        Value::Num(total / cnt as f64)
                    }
                }
                _ => best.map_or(Value::Null, Value::Num), // min/max of nothing → NULL
            };
            Col::Gen(vec![v])
        })
        .collect();
    Some(Batch::of(cols))
}

/// Try to answer a scalar `count(*)` / `count(DISTINCT <last slot>)` sitting on
/// an Expand of a Scan/Expand chain WITHOUT materializing the wide intermediate
/// batch: the frontier feeding the final hop is produced by [`frontier_ids`],
/// then `count(*)` sums the final hop's matching degree and `count(DISTINCT c)`
/// marks endpoints in a bitset over node ids. Returns `None` (fall back to the
/// general aggregate) for any shape it does not recognize — so it is an
/// optimization, never a semantic fork.
/// Peel exactly `n` OUTgoing frontier hops (no bound edge) ending at a bare Scan,
/// returning the per-hop edge labels FIRST-to-LAST and the Scan's label. `None`
/// unless the plan is precisely that chain (used by the 3-hop edge-product count).
fn peel_out_hops(plan: &Plan, n: usize) -> Option<(Vec<Vec<String>>, Option<String>)> {
    if n == 0 {
        return match plan {
            Plan::Scan { label } => Some((Vec::new(), label.clone())),
            _ => None,
        };
    }
    let Plan::Expand {
        input,
        from,
        dir: Dir::Out,
        edge_label,
        bind_edge: false,
        double_loops: false,
    } = plan
    else {
        return None;
    };
    if *from + 1 != chain_width(input)? {
        return None; // must expand the current frontier
    }
    let (mut labels, base) = peel_out_hops(input, n - 1)?;
    labels.push(edge_label.clone());
    Some((labels, base))
}

/// count(*) over a 3-hop OUT chain via the identity `1ᵀA₁A₂A₃1 = Σ` over the MIDDLE
/// edges (b→c, hop 2) of `(source→b walks over hop 1) × (out-degree of c over hop
/// 3)` — O(V+E), replacing the 2-hop count-propagation SCATTER (the 3-hop
/// bottleneck: random `next[nbr] += c` writes) with degree products. A fixed chain
/// is a WALK (edges may repeat), so there is NO trail correction — byte-identical
/// to the propagation. Per-hop edge types are handled independently.
fn try_3hop_product_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None;
    }
    let (labels, base) = peel_out_hops(input, 3)?;
    let mut wants: Vec<Vec<u32>> = Vec::with_capacity(3);
    for l in &labels {
        match want_etypes(store, l) {
            Ok(w) => wants.push(w),
            Err(()) => return Some(scalar_num(0.0)), // unknown edge type → no paths
        }
    }
    let (w1, w2, w3) = (&wants[0], &wants[1], &wants[2]);
    let nc = store.node_count();
    // Empty want = any type; else the edge must carry one of the hop's labels
    // (primary or, on a multi-label graph, secondary).
    let hit = |a: &crate::store::Adj, w: &[u32]| edge_carries_wanted(store, a, w);

    // level1[b] = number of hop-1 edges from a SOURCE into b (= counts after 1 hop).
    let mut level1 = vec![0u64; nc];
    let bump = |s: u32, level1: &mut [u64]| {
        for a in store.out(s) {
            if hit(a, w1) {
                level1[a.nbr as usize] += 1;
            }
        }
    };
    match &base {
        Some(l) => {
            for &s in store.nodes_with_label(l) {
                bump(s, &mut level1);
            }
        }
        None => {
            for s in 0..nc as u32 {
                if store.is_alive(s) {
                    bump(s, &mut level1);
                }
            }
        }
    }
    // outdeg3[c] = number of hop-3 out-edges of c.
    let mut outdeg3 = vec![0u64; nc];
    for (c, d) in outdeg3.iter_mut().enumerate() {
        *d = store.out(c as u32).iter().filter(|a| hit(a, w3)).count() as u64;
    }
    // Σ over hop-2 middle edges (b→c) of level1[b] × outdeg3[c].
    let mut total = 0u64;
    for (b, &lvl) in level1.iter().enumerate() {
        if lvl == 0 {
            continue;
        }
        for a in store.out(b as u32) {
            if hit(a, w2) {
                total += lvl * outdeg3[a.nbr as usize];
            }
        }
    }
    Some(scalar_num(total as f64))
}

fn try_fused_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count {
        return None;
    }
    let Plan::Expand {
        input: inner,
        from,
        dir,
        edge_label,
        double_loops,
        ..
    } = input
    else {
        return None;
    };
    // Gremlin `both()` walks a self-loop twice — the final-hop degree counts it twice.
    let dl = *double_loops;
    let w = chain_width(inner)?; // slot count feeding the final hop
    if *from + 1 != w {
        return None; // the final Expand must expand the current frontier
    }
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)), // unknown label: zero rows
    };
    let src = frontier_ids(inner, store)?; // ids feeding the final hop, w/ multiplicity

    if agg.arg.is_none() {
        // DEEP chain (≥2 hops feed the final hop, so the intermediate frontier would
        // explode with path multiplicity): propagate a per-node count array instead
        // of materializing the frontier ids — O(hops * edges) time, O(node_count)
        // space. The count is Σ_v counts[v] * matching-out-degree(v).
        if count_hops(inner) >= 2 {
            if let Some(counts) = frontier_counts(inner, store) {
                let mut total = 0f64;
                counts.for_each(|v, c| {
                    let mut deg = 0f64;
                    for_each_nbr(store, v, *dir, &want, dl, |_, _| deg += 1.0);
                    total += c * deg;
                });
                return Some(scalar_num(total));
            }
        }
        // count(*): number of final-hop paths = sum over sources of matching
        // out-degree. When the sources come from an Expand they repeat (many paths
        // reach the same node), and a node's degree is the same each time — so
        // collapse to distinct nodes with multiplicity and walk each adjacency
        // once, scaled. When they come from a Scan they are already distinct, so
        // that dedup is pure overhead: sum degrees directly.
        //
        // When the hop's type set matches EVERY edge (an unlabeled hop, or the
        // graph's only type), the degree is the raw adjacency length — no per-edge
        // type check. That is the common "count all my out-neighbours" shape; the
        // per-edge walk it replaces was the one 1-hop-count regression vs core.
        let all_types = want_covers_all_etypes(store, &want);
        let mut total = 0f64;
        if matches!(inner.as_ref(), Plan::Expand { .. }) {
            let (distinct, mult) = distinct_with_mult(&src, store.node_count());
            for (i, &v) in distinct.iter().enumerate() {
                total += mult[i] * matching_degree(store, v, *dir, &want, dl, all_types);
            }
        } else {
            for &v in &src {
                total += matching_degree(store, v, *dir, &want, dl, all_types);
            }
        }
        return Some(scalar_num(total));
    }
    if agg.distinct {
        // count(DISTINCT c) where c is the final (last) slot, index == w: distinct
        // endpoints deduped in a bitset — no per-row hashing, no boxed values.
        match agg.arg.as_ref() {
            Some(Expr::Slot(s)) if *s == w => {}
            _ => return None,
        }
        // The distinct endpoints depend only on the SET of last-hop sources, not
        // their multiplicity: a source reached by many paths yields the same
        // neighbours each time. When the sources come from an Expand they repeat,
        // so collapse them to distinct nodes first — a 2-hop's millions of repeated
        // intermediates down to the distinct nodes, each final hop walked once.
        // Sources from a Scan are already distinct, so skip that pass.
        let nc = store.node_count();
        let deduped;
        let sources: &[u32] = if matches!(inner.as_ref(), Plan::Expand { .. }) {
            let mut seen_src = vec![false; nc];
            let mut distinct_src = Vec::new();
            for &v in &src {
                if !seen_src[v as usize] {
                    seen_src[v as usize] = true;
                    distinct_src.push(v);
                }
            }
            deduped = distinct_src;
            &deduped
        } else {
            &src
        };
        let mut seen = vec![false; nc];
        let mut cnt = 0f64;
        for &v in sources {
            for_each_nbr(store, v, *dir, &want, false, |nbr, _| {
                if !seen[nbr as usize] {
                    seen[nbr as usize] = true;
                    cnt += 1.0;
                }
            });
        }
        return Some(scalar_num(cnt));
    }
    None // count(arg) non-distinct on the final slot: not fused (uncommon)
}

/// A one-row, one-column batch holding a single number — a scalar aggregate's
/// result.
fn scalar_num(x: f64) -> Batch {
    Batch::of(vec![Col::Gen(vec![Value::Num(x)])])
}

/// A Gremlin `tree()` accumulator: a nested, INSERTION-ORDERED map (children keyed by
/// value, matched via `value::equals`), materialized into nested `Value::Map`s.
#[derive(Default)]
struct GremlinTree {
    // Children in FIRST-SEEN order (the Gremlin tree contract) …
    order: Vec<(Value, GremlinTree)>,
    // … plus a grouping-key → index map so a level with many children is an O(1) hash
    // lookup, not a linear scan comparing full element-map keys (which made a wide
    // tree O(paths · children · map-size) — the dominant `tree()` cost).
    index: FnvMap<Vec<u8>, usize>,
}

impl GremlinTree {
    fn insert(&mut self, keys: &[Value]) {
        let Some((first, rest)) = keys.split_first() else {
            return;
        };
        let mut kb = Vec::new();
        crate::value::group_key_into(first, &mut kb);
        let i = match self.index.get(&kb) {
            Some(&i) => i,
            None => {
                let idx = self.order.len();
                self.order.push((first.clone(), GremlinTree::default()));
                self.index.insert(kb, idx);
                idx
            }
        };
        self.order[i].1.insert(rest);
    }

    fn to_value(&self) -> Value {
        Value::Map(std::sync::Arc::new(
            self.order
                .iter()
                .map(|(k, c)| (k.clone(), c.to_value()))
                .collect(),
        ))
    }
}

/// A component/cluster id as the ROOT vertex's external-id string — the value core's
/// `connectedComponent`/`peerPressure` write (`Value::Str(vid.arc(root))`). A root
/// with no external id (never, for a loaded node) reads back NULL.
fn root_ext_id(store: &Store, root: u32) -> Value {
    store.node_ext_id(root).map_or(Value::Null, Value::Str)
}

/// Collapse a node-id multiset to (distinct ids in first-seen order, their
/// multiplicities) via a direct-mapped array — node ids are dense, so no hashing.
fn distinct_with_mult(nodes: &[u32], node_count_total: usize) -> (Vec<u32>, Vec<f64>) {
    let mut group_of = vec![u32::MAX; node_count_total];
    let mut distinct: Vec<u32> = Vec::new();
    let mut mult: Vec<f64> = Vec::new();
    for &id in nodes {
        let slot = &mut group_of[id as usize];
        if *slot == u32::MAX {
            *slot = u32::try_from(distinct.len()).expect("distinct count fits in u32");
            distinct.push(id);
            mult.push(1.0);
        } else {
            mult[*slot as usize] += 1.0;
        }
    }
    (distinct, mult)
}

/// Does `expr` reference no slot other than `s` (and never the path)? Literals
/// and comparisons over slot `s` qualify; any other slot, or `Expr::Path`,
/// disqualifies — the signal that the frontier alone is enough to evaluate it.
fn refs_only_slot(expr: &Expr, s: usize) -> bool {
    match expr {
        Expr::Lit(_) | Expr::Param(_) => true,
        Expr::Slot(n) => *n == s,
        Expr::Prop { slot, .. } => *slot == s,
        Expr::Path | Expr::PathAccess { .. } | Expr::GremlinPath { .. } => false,
        Expr::Not(x) => refs_only_slot(x, s),
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Xor(a, b)
        | Expr::Arith {
            left: a, right: b, ..
        }
        | Expr::In {
            needle: a,
            haystack: b,
        } => refs_only_slot(a, s) && refs_only_slot(b, s),
        Expr::Call { args, .. } | Expr::GraphPred { args, .. } | Expr::List { items: args } => {
            args.iter().all(|a| refs_only_slot(a, s))
        }
        Expr::Record { fields } | Expr::MapLit { entries: fields } => {
            fields.iter().all(|(_, e)| refs_only_slot(e, s))
        }
        Expr::Field { base, .. } => refs_only_slot(base, s),
        Expr::Index { base, index, .. } => refs_only_slot(base, s) && refs_only_slot(index, s),
        Expr::Case {
            branches,
            otherwise,
        } => {
            branches
                .iter()
                .all(|(c, v)| refs_only_slot(c, s) && refs_only_slot(v, s))
                && otherwise.as_deref().is_none_or(|e| refs_only_slot(e, s))
        }
        Expr::Compare { left, right, .. } => refs_only_slot(left, s) && refs_only_slot(right, s),
        Expr::Cast { expr, .. } | Expr::IsNull { expr, .. } => refs_only_slot(expr, s),
        Expr::PropertyExists { slot, .. } | Expr::IsLabeled { slot, .. } => *slot == s,
        // An EXISTS correlates on outer slots below `outer_width`; conservatively
        // treat it as touching more than one, so it never rides the frontier-only
        // aggregate fast path.
        Expr::Exists { .. }
        | Expr::CountSubquery { .. }
        | Expr::ScalarSubquery { .. }
        | Expr::CollectSubquery { .. }
        | Expr::UncorrelatedExists { .. }
        | Expr::UncorrelatedCount { .. }
        | Expr::UncorrelatedScalar { .. } => false,
    }
}

/// Rewrite every reference to slot `from` in `expr` to slot `to`. Used to retarget
/// frontier-only expressions onto a one-slot frontier batch. Callers guarantee
/// (via [`refs_only_slot`]) that no other slot appears.
fn remap_slot(expr: &Expr, from: usize, to: usize) -> Expr {
    let go = |e| Box::new(remap_slot(e, from, to));
    match expr {
        Expr::Slot(n) if *n == from => Expr::Slot(to),
        Expr::Prop { slot, key } if *slot == from => Expr::Prop {
            slot: to,
            key: key.clone(),
        },
        Expr::Slot(_)
        | Expr::Prop { .. }
        | Expr::Lit(_)
        | Expr::Path
        | Expr::PathAccess { .. }
        | Expr::GremlinPath { .. } => expr.clone(),
        Expr::Not(x) => Expr::Not(go(x)),
        Expr::And(a, b) => Expr::And(go(a), go(b)),
        Expr::Or(a, b) => Expr::Or(go(a), go(b)),
        Expr::Xor(a, b) => Expr::Xor(go(a), go(b)),
        Expr::In { needle, haystack } => Expr::In {
            needle: go(needle),
            haystack: go(haystack),
        },
        Expr::Compare { op, left, right } => Expr::Compare {
            op: *op,
            left: go(left),
            right: go(right),
        },
        Expr::Arith { op, left, right } => Expr::Arith {
            op: *op,
            left: go(left),
            right: go(right),
        },
        Expr::Call { name, args } => Expr::Call {
            name: name.clone(),
            args: args.iter().map(|a| remap_slot(a, from, to)).collect(),
        },
        Expr::GraphPred { op, args, negated } => Expr::GraphPred {
            op: *op,
            args: args.iter().map(|a| remap_slot(a, from, to)).collect(),
            negated: *negated,
        },
        Expr::List { items } => Expr::List {
            items: items.iter().map(|a| remap_slot(a, from, to)).collect(),
        },
        Expr::Record { fields } => Expr::Record {
            fields: fields
                .iter()
                .map(|(k, e)| (k.clone(), remap_slot(e, from, to)))
                .collect(),
        },
        Expr::MapLit { entries } => Expr::MapLit {
            entries: entries
                .iter()
                .map(|(k, e)| (k.clone(), remap_slot(e, from, to)))
                .collect(),
        },
        Expr::Index { base, index, elem } => Expr::Index {
            base: go(base),
            index: go(index),
            elem: *elem,
        },
        Expr::Field { base, key } => Expr::Field {
            base: go(base),
            key: key.clone(),
        },
        Expr::Case {
            branches,
            otherwise,
        } => Expr::Case {
            branches: branches
                .iter()
                .map(|(c, v)| (remap_slot(c, from, to), remap_slot(v, from, to)))
                .collect(),
            otherwise: otherwise
                .as_ref()
                .map(|e| Box::new(remap_slot(e, from, to))),
        },
        Expr::Cast { target, expr } => Expr::Cast {
            target: *target,
            expr: go(expr),
        },
        Expr::IsNull { expr, negated } => Expr::IsNull {
            expr: go(expr),
            negated: *negated,
        },
        Expr::PropertyExists { slot, key } => Expr::PropertyExists {
            slot: if *slot == from { to } else { *slot },
            key: key.clone(),
        },
        Expr::IsLabeled { slot, labels } => Expr::IsLabeled {
            slot: if *slot == from { to } else { *slot },
            labels: labels.clone(),
        },
        // Never reached: `refs_only_slot` rejects EXISTS, so the frontier remap
        // that calls this is never handed one. Clone rather than rewrite a body.
        Expr::Exists { .. }
        | Expr::CountSubquery { .. }
        | Expr::ScalarSubquery { .. }
        | Expr::CollectSubquery { .. }
        | Expr::UncorrelatedExists { .. }
        | Expr::UncorrelatedCount { .. }
        | Expr::UncorrelatedScalar { .. }
        | Expr::Param(_) => expr.clone(),
    }
}

/// `count(*)` grouped by a single property of the frontier node, computed by
/// grouping on the integer node id FIRST, then merging node groups by the
/// property value. The property is a function of the node, so two rows on the
/// same node share a property value: counting 8M endpoints by their (cheap,
/// dense) node id and reading/hashing the property for only the distinct nodes
/// replaces millions of string hashes and `Arc` clones with a direct-mapped
/// array index each. The final hop is fused into the count — endpoints are
/// streamed straight into the array, never materialized as a column. First-seen
/// order is preserved: the distinct nodes are visited in first-appearance order,
/// so a property value is first seen at the earliest node — hence earliest row —
/// carrying it. `None` for any other shape (non-count aggregate, key that is not
/// a lone frontier property), which falls through to the general frontier path.
///
/// Rejected optimization: for a DICT-encoded key, counting straight into per-code
/// buckets during the traversal (`counts[codes[nbr]] += 1`), skipping this per-node
/// intermediate and the Level-2 merge. It moved `c.city, count(*)` on the 2-hop
/// 100k/deg-5 fixture only 24.5ms -> 23.0ms (0.54x -> 0.57x of core) — a consistent
/// ~7% but still far from parity, and it TRADES the per-node scatter for reading the
/// property once PER PATH (2.5M reads) instead of once per distinct endpoint (100k).
/// The shape is memory-bound on ~2.5M random accesses either way; core's remaining
/// edge is its CSR adjacency (sequential neighbour reads), which the per-node `Vec`
/// adjacency here cannot match without a layout change (deferred, large blast radius).
/// Not worth a second grouped-count path for a sub-10% move that leaves it slowest.
fn try_node_grouped_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    let [agg] = aggs else { return None };
    if agg.func != AggFn::Count || agg.arg.is_some() {
        return None;
    }
    let [(_, key_expr)] = keys else { return None };
    // The group node is the endpoint of a final Expand over a Scan/Expand chain.
    let Plan::Expand {
        input: inner,
        from,
        dir,
        edge_label,
        bind_edge,
        double_loops: false,
    } = input
    else {
        return None;
    };
    if *bind_edge {
        // With the edge bound, the endpoint node sits at slot w+1 and slot w is the
        // EDGE — so a `Prop{slot: w}` key is an edge property, not a node one. This
        // fast path reads NODE properties of the endpoint; hand a bound-edge group
        // (e.g. `RETURN r.w, count(*)`) to the general aggregate, which reads the
        // edge slot correctly. (Found by the differential fuzzer: this used to read
        // the edge key as an absent node property and bucket every row under NULL.)
        return None;
    }
    let w = chain_width(inner)?;
    if *from + 1 != w {
        return None; // the final Expand must expand the current frontier
    }
    let Expr::Prop { slot, key } = key_expr else {
        return None;
    };
    if *slot != w {
        return None; // key must read the endpoint (last) slot, index == w
    }
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(Batch::of(vec![Col::Nodes(vec![]), Col::Gen(vec![])])),
    };
    let src = frontier_ids(inner, store)?; // nodes feeding the final hop, w/ multiplicity

    // Level 1: count per endpoint node id via a direct-mapped array (no hashing —
    // node ids are dense), with the final hop fused in so endpoints never
    // materialize. Distinct ids come out in first-seen order.
    let mut group_of = vec![u32::MAX; store.node_count()];
    let mut rep_ids: Vec<u32> = Vec::new();
    let mut node_count: Vec<f64> = Vec::new();
    for &v in &src {
        for_each_nbr(store, v, *dir, &want, false, |nbr, _| {
            let slot = &mut group_of[nbr as usize];
            if *slot == u32::MAX {
                *slot = u32::try_from(rep_ids.len()).expect("group count fits in u32");
                rep_ids.push(nbr);
                node_count.push(1.0);
            } else {
                node_count[*slot as usize] += 1.0;
            }
        });
    }

    // Read the grouping property for the DISTINCT endpoint nodes only.
    let key_col = read_property(store, &Col::Nodes(rep_ids), key);

    // Level 2: merge node groups by property value, summing their counts.
    let (val_of, val_first) = assign_groups(std::slice::from_ref(&key_col), key_col.len());
    let mut counts = vec![0f64; val_first.len()];
    for (node_group, &vg) in val_of.iter().enumerate() {
        counts[vg as usize] += node_count[node_group];
    }
    let key_out = key_col.gather(&val_first);
    Some(Batch::of(vec![
        key_out,
        Col::Gen(counts.into_iter().map(Value::Num).collect()),
    ]))
}

/// Run a grouped/scalar aggregate over a Scan/Expand chain WITHOUT materializing
/// the earlier slots: when every key and aggregate argument reads only the
/// frontier (last) slot, the chain's frontier is all the aggregate needs. The
/// frontier ([`frontier_ids`]) is produced in the same row order the full batch
/// would have, so first-seen group order — and every value — is identical to the
/// general path; this only drops the wasted slot columns. `None` for any shape it
/// does not handle (a filter/join in the chain, an expression over an earlier
/// slot), which falls back to the general aggregate.
fn try_frontier_aggregate(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Result<Option<Batch>, String> {
    let Some(width) = chain_width(input) else {
        return Ok(None);
    };
    let last = width - 1; // frontier slot index of the whole chain
    let key_ok = keys.iter().all(|(_, e)| refs_only_slot(e, last));
    let agg_ok = aggs
        .iter()
        .all(|a| a.arg.as_ref().is_none_or(|e| refs_only_slot(e, last)));
    if !key_ok || !agg_ok {
        return Ok(None);
    }
    let Some(frontier) = frontier_ids(input, store) else {
        return Ok(None);
    };
    let batch = Batch::of(vec![Col::Nodes(frontier)]);
    // Retarget the frontier-only expressions onto the one-slot frontier batch.
    let keys: Vec<(String, Expr)> = keys
        .iter()
        .map(|(n, e)| (n.clone(), remap_slot(e, last, 0)))
        .collect();
    let aggs: Vec<Agg> = aggs
        .iter()
        .map(|a| Agg {
            func: a.func,
            arg: a.arg.as_ref().map(|e| remap_slot(e, last, 0)),
            distinct: a.distinct,
            name: a.name.clone(),
            frac: a.frac,
            null_on_empty: a.null_on_empty,
        })
        .collect();
    Ok(Some(aggregate(&batch, store, &keys, &aggs)?))
}

/// A quantified hop: for each input row, enumerate every path of length in
/// `min..=max` from the node in `from`, and emit one output row per path with the
/// reached endpoint appended as a new slot. `min == 0` emits the source itself.
///
/// `mode` chooses the semantics and nothing else does (see [`PathMode`]): `Trail`
/// forbids reusing an edge, `Walk` allows anything, `Simple`/`Acyclic` forbid
/// reusing a node (Simple permits the closing `start == end`). They diverge on a
/// cycle/self-loop — pinned by the tests — and are never conflated with a chain
/// of separate fixed `Expand`s (which is always a walk).
#[allow(clippy::too_many_arguments)]
fn var_length(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: &[String],
    min: u32,
    max: u32,
    mode: PathMode,
    // A quantified subpath group binds inner variables as GROUP lists (RepeatGroup);
    // each `(pos, _slot)` appends one list column (source/edge/target per rep) after
    // the endpoint. Empty = a plain var-length hop (endpoint only).
    group_binds: &[(crate::ir::GroupPos, usize)],
    // A per-repetition `WHERE` (RepeatGroup) over the rep's SCALAR variables at fixed
    // mini-scope slots (source=0, edge=1, target=2); a hop failing it is pruned.
    per_rep_pred: Option<&Expr>,
    // Hops per repetition unit: an endpoint is emitted only at a rep boundary
    // (`len % k == 0`). 1 for a plain hop or a single-hop group.
    k: u32,
    // Gremlin `until(pred)`: emit an endpoint ONLY when `pred` holds (evaluated over a
    // one-row mini-batch whose endpoint slot carries the landed node) and PRUNE that
    // branch on a match. `min` decides the earliest depth checked (0 pre-form, 1 post).
    until_stop: Option<&Expr>,
    // Gremlin `repeat(<hop>.<filter>)` body filter: a predicate on each hop TARGET
    // (`len > 0`); a target that fails it is pruned (no emit, no descent).
    body_filter: Option<&Expr>,
    // Gremlin `both()` self-loop doubling — a self-loop is walked twice (see the
    // `double_loops` field on Plan::VarLength).
    double_loops: bool,
) -> Result<Batch, String> {
    // Anti-runaway budget: a per-path (WALK) `repeat`/var-length expansion fans out as
    // degree^depth, so on a dense graph it materializes billions of rows and OOM-kills
    // the host. Cap the total emitted rows at `limits.trail` (core's guard, same
    // default); the DFS stops descending once past it and we return a loud
    // `E_RESOURCE_EXHAUSTED` rather than a truncated result.
    let budget = store.limits().trail;
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        for _ in group_binds {
            slots.push(Col::Gen(vec![]));
        }
        Batch::of(slots)
    };
    // An unknown edge type matches NO edge, but a `*`/`{0,…}` still emits each source
    // at zero reps (endpoint = source, group lists empty). A never-matching set (NOT
    // the empty "any" set) lets the DFS traverse nothing yet still emit those.
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => vec![u32::MAX],
    };
    // The walk appends its endpoint at the first free slot; an `until(pred)` mini-batch
    // places the landed node there so a pred like `hasLabel(endpoint)` resolves.
    let endpoint_slot = batch.slots.len();
    let Col::Nodes(src) = batch.slot(from) else {
        return Ok(empty());
    };

    // A named path over the pattern (`MATCH p = (a)-[:R]->{1,3}(b)`) needs the
    // per-row node/edge chain so path_length(p)/nodes(p)/edges(p) resolve; the
    // input carries a lineage exactly when the plan reads the path.
    let track = batch.lineage.is_some();
    // Materialize each emitted path into keep/ends (+ lineage / group columns).
    let mut sink = CollectEmit {
        keep: Vec::new(),
        ends: Vec::new(),
        bufs: PathBufs::new(),
        group_cols: vec![Vec::new(); group_binds.len()],
        batch: if track { Some(batch) } else { None },
        group_binds,
        k,
        budget,
    };
    run_varlen(
        src,
        store,
        &want,
        min,
        max,
        dir,
        mode,
        per_rep_pred,
        k,
        until_stop.map(|p| (p, endpoint_slot)),
        body_filter.map(|p| (p, endpoint_slot)),
        double_loops,
        &mut sink,
    );

    if sink.keep.len() as u64 > budget {
        return Err(format!(
            "E_RESOURCE_EXHAUSTED: variable-length traversal exceeded the trail limit of \
             {budget} rows; add a tighter bound/`LIMIT`, dedup the frontier, or raise the \
             limit (ConfigId::LimitsTrail)"
        ));
    }
    let CollectEmit {
        keep,
        ends,
        bufs,
        group_cols,
        ..
    } = sink;

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(ends));
    for col in group_cols {
        slots.push(Col::Gen(col));
    }
    let mut out = Batch::of(slots);
    if track {
        out.lineage = Some(Lineage {
            values: bufs.values,
            offsets: bufs.offsets,
            edges: bufs.edges,
            edge_offsets: bufs.edge_offsets,
        });
    }
    Ok(out)
}

/// Materialize the group-variable lists for ONE emitted repetition-path and push
/// each into its column. `node_stack` = `[source, …, endpoint]` (len `reps*k + 1`),
/// `edge_stack` the flat edges (len `reps*k`); `k` is the unit hop count. The
/// variable at `NodeAt(p)` collects `node_stack[rep*k + p]` across reps, `EdgeAt(p)`
/// collects `edge_stack[rep*k + p]` — each a `Value::Num` id inside a `Value::List`.
/// Evaluate a per-repetition `WHERE` over the rep that ENDS at hop `len` (its `k`
/// hops occupy `edge_stack[len-k..len]` and its `k+1` nodes `node_stack[len-k..=len]`).
/// Binds node position `p` at mini-scope slot `2p`, edge position `p` at `2p+1`, then
/// evaluates over that one-row batch. A false / null / faulting predicate fails the rep.
fn rep_pred_ok(
    pred: &Expr,
    store: &Store,
    node_stack: &[u32],
    edge_stack: &[u32],
    len: u32,
    k: u32,
) -> bool {
    let (len, k) = (len as usize, k as usize);
    let base = len - k;
    let mut slots: Vec<Col> = Vec::with_capacity(2 * k + 2);
    for p in 0..=k {
        slots.push(Col::Nodes(vec![node_stack[base + p]]));
        if p < k {
            slots.push(Col::Edges(vec![edge_stack[base + p]]));
        }
    }
    // Slot `2k+1` carries the path SOURCE, so a per-hop WHERE can reference the hop's
    // outer anchor variable (`(a)-[e WHERE a.k = …]->{…}`); the parser maps that
    // variable to this slot. `node_stack[0]` is the source for every repetition.
    slots.push(Col::Nodes(vec![node_stack[0]]));
    let mini = Batch::of(slots);
    eval(pred, store, &mini)
        .map(|c| c.value_at(0).is_true())
        .unwrap_or(false)
}

/// Evaluate a Gremlin `until(pred)` at a single walk endpoint `v`. The predicate was
/// built referencing the endpoint slot, so a one-row mini-batch places `v` there (and
/// at every lower slot, harmless — a well-formed until pred reads only the endpoint).
fn until_ok(pred: &Expr, store: &Store, endpoint_slot: usize, v: u32) -> bool {
    // Fast path: the common until/body-filter predicates (hasLabel, hasProp, and their
    // boolean combinations) evaluate DIRECTLY on the node — no per-node Batch allocation.
    // Anything else (a value comparison, a nested Exists) falls back to the mini-batch.
    if let Some(b) = eval_node_bool(pred, store, v) {
        return b;
    }
    let slots: Vec<Col> = (0..=endpoint_slot).map(|_| Col::Nodes(vec![v])).collect();
    let mini = Batch::of(slots);
    eval(pred, store, &mini)
        .map(|c| c.value_at(0).is_true())
        .unwrap_or(false)
}

/// Evaluate a boolean predicate directly against a single node `v`, for the EXACT,
/// allocation-free forms a repeat `until`/body-filter uses (all slots in the mini-batch
/// are `v`, so the slot index is immaterial). `None` = a form this can't handle exactly
/// (`Compare`, `Case`, `Exists`, …) — the caller falls back to the batch evaluator.
fn eval_node_bool(pred: &Expr, store: &Store, v: u32) -> Option<bool> {
    match pred {
        Expr::IsLabeled { labels, .. } => Some(labels.iter().any(|l| store.is_labeled(v, l))),
        Expr::PropertyExists { key, .. } => Some(store.has_prop(v, key)),
        Expr::Lit(Value::Bool(b)) => Some(*b),
        Expr::Not(x) => eval_node_bool(x, store, v).map(|b| !b),
        Expr::And(a, b) => match (eval_node_bool(a, store, v), eval_node_bool(b, store, v)) {
            (Some(x), Some(y)) => Some(x && y),
            _ => None,
        },
        Expr::Or(a, b) => match (eval_node_bool(a, store, v), eval_node_bool(b, store, v)) {
            (Some(x), Some(y)) => Some(x || y),
            _ => None,
        },
        _ => None,
    }
}

fn push_group_cols(
    node_stack: &[u32],
    edge_stack: &[u32],
    k: u32,
    group_binds: &[(crate::ir::GroupPos, usize)],
    group_cols: &mut [Vec<Value>],
) {
    use crate::ir::GroupPos;
    let k = k as usize;
    let reps = edge_stack.len() / k;
    for (i, (pos, _)) in group_binds.iter().enumerate() {
        let list: Vec<Value> = match pos {
            GroupPos::NodeAt(p) => (0..reps)
                .map(|r| Value::Num(f64::from(node_stack[r * k + *p as usize])))
                .collect(),
            GroupPos::EdgeAt(p) => (0..reps)
                .map(|r| Value::Num(f64::from(edge_stack[r * k + *p as usize])))
                .collect(),
        };
        group_cols[i].push(Value::List(list));
    }
}

// ── NESTED subpath groups (`Plan::NestedGroup`) ──────────────────────────────

/// One graph-consuming hop of a matched trail, tagged with its position in the
/// (nested) repetition pattern. `levels` is the cursor stack outer→inner: one
/// `(rep, elem_after)` per active unit — `elem_after` is the element index the hop
/// advanced PAST, EXCEPT that a step inside a `Sub` keeps the enclosing unit's entry
/// pinned at that Sub's element index. This is what lets the structured binder place
/// each variable at the right nesting depth. Mirrors core's `pathfind::StepRec`.
#[derive(Clone)]
struct StepRec {
    levels: Vec<(u32, usize)>,
    source: u32,
    edge: u32,
    target: u32,
}

/// A partially-built nested list keyed by a rep-tuple: `insert([i,j], v)` puts `v` at
/// `list[i][j]`, growing intermediate lists. Depth-`d` variable → `d+1`-element keys.
enum Nest {
    Leaf(Value),
    List(Vec<Nest>),
}
impl Nest {
    fn insert(&mut self, idx: &[u32], val: Value) {
        match idx.split_first() {
            None => *self = Nest::Leaf(val),
            Some((&i, rest)) => {
                if !matches!(self, Nest::List(_)) {
                    *self = Nest::List(Vec::new());
                }
                if let Nest::List(v) = self {
                    let i = i as usize;
                    while v.len() <= i {
                        v.push(Nest::List(Vec::new()));
                    }
                    v[i].insert(rest, val);
                }
            }
        }
    }
    fn into_val(self) -> Value {
        match self {
            Nest::Leaf(v) => v,
            Nest::List(items) => Value::List(items.into_iter().map(Nest::into_val).collect()),
        }
    }
}

/// Assemble every bound variable of `unit` (recursively into its `Sub`s) as a
/// (possibly nested) list keyed by the repetition counters of the units it sits in —
/// one list level per enclosing quantifier. `tree_path` is the `Sub`-element indices
/// from the top unit to THIS one, so `depth = tree_path.len()` is its nesting depth.
/// A node/edge id is stored as `Value::Num(id)` (the group-variable convention; the
/// `x[i].prop` element-typing reads it back). Mirrors core's `pathfind::bind_unit`
/// with `key_start = 0`.
fn bind_nested(
    unit: &crate::ir::GUnit,
    tree_path: &[usize],
    key_start: usize,
    steps: &[StepRec],
    out: &mut Vec<(usize, Value)>,
) {
    use crate::ir::GElem;
    let depth = tree_path.len();
    // `key_start = 0` = the full-nesting emit view; `key_start = 1` drops the outer-rep
    // index for a PER-REP `WHERE` (each var one level shallower). Clamp to `depth+1`.
    let ks = key_start.min(depth + 1);
    let key = |s: &StepRec| -> Vec<u32> { s.levels[ks..=depth].iter().map(|(r, _)| *r).collect() };
    let within = |s: &StepRec| -> bool {
        s.levels.len() > depth
            && s.levels[..depth]
                .iter()
                .map(|(_, e)| *e)
                .eq(tree_path.iter().copied())
    };
    // The unit's source = each rep-instance's FIRST hop's source (deduped per key).
    if let Some(slot) = unit.start_slot {
        let mut nest = Nest::List(Vec::new());
        let mut seen: std::collections::HashSet<Vec<u32>> = std::collections::HashSet::new();
        for s in steps.iter().filter(|s| within(s)) {
            let k = key(s);
            if seen.insert(k.clone()) {
                nest.insert(&k, Value::Num(f64::from(s.source)));
            }
        }
        out.push((slot, nest.into_val()));
    }
    for (e, elem) in unit.elems.iter().enumerate() {
        match elem {
            GElem::Hop {
                edge_slot,
                target_slot,
                ..
            } => {
                let direct = |s: &&StepRec| {
                    within(s) && s.levels.len() == depth + 1 && s.levels[depth].1 == e + 1
                };
                if let Some(slot) = target_slot {
                    let mut nest = Nest::List(Vec::new());
                    for s in steps.iter().filter(direct) {
                        nest.insert(&key(s), Value::Num(f64::from(s.target)));
                    }
                    out.push((*slot, nest.into_val()));
                }
                if let Some(slot) = edge_slot {
                    let mut nest = Nest::List(Vec::new());
                    for s in steps.iter().filter(direct) {
                        nest.insert(&key(s), Value::Num(f64::from(s.edge)));
                    }
                    out.push((*slot, nest.into_val()));
                }
            }
            GElem::Sub {
                unit: sub,
                target_slot,
                ..
            } => {
                // The Sub's landing = its LAST inner hop's target, per rep-instance.
                if let Some(slot) = target_slot {
                    let mut last: Vec<(Vec<u32>, u32)> = Vec::new();
                    for s in steps.iter().filter(|s| {
                        within(s) && s.levels.len() > depth + 1 && s.levels[depth].1 == e
                    }) {
                        let k = key(s);
                        match last.iter_mut().find(|(kk, _)| *kk == k) {
                            Some(slot) => slot.1 = s.target,
                            None => last.push((k, s.target)),
                        }
                    }
                    let mut nest = Nest::List(Vec::new());
                    for (k, t) in last {
                        nest.insert(&k, Value::Num(f64::from(t)));
                    }
                    out.push((*slot, nest.into_val()));
                }
                let mut child = tree_path.to_vec();
                child.push(e);
                bind_nested(sub, &child, key_start, steps, out);
            }
        }
    }
}

/// `Plan::NestedGroup`: a subpath group `( <unit> ){min,max}` whose body is a single
/// nested quantified sub-group / quantified inner hop (the 2-level shape the corpus
/// and fuzzer produce: `( ((x)-[e]->(y)){a,b} ){c,d}` and `( (x)-[e]->{a,b}(y)
/// ){c,d}`). Enumerates every valid outer×inner repetition-decomposition as a TRAIL
/// and materializes each bound inner variable as a (nested) list via `bind_nested`.
#[allow(clippy::too_many_arguments)]
fn nested_group(
    batch: &Batch,
    store: &Store,
    from: usize,
    unit: &crate::ir::GUnit,
    min: u32,
    max: u32,
    mode: PathMode,
    bind_slots: &[usize],
    per_rep_pred: Option<&Expr>,
) -> Batch {
    use crate::ir::GElem;
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        for _ in bind_slots {
            slots.push(Col::Gen(vec![]));
        }
        Batch::of(slots)
    };
    // Each outer element is a Hop or a Sub whose inner unit is FLAT (hops only — no
    // deeper than 2 levels). Anything else is unsupported here → no rows.
    for el in &unit.elems {
        if let GElem::Sub { unit: sub, .. } = el {
            if sub.elems.iter().any(|e| matches!(e, GElem::Sub { .. })) {
                return empty();
            }
        }
    }
    let Col::Nodes(src) = batch.slot(from) else {
        return empty();
    };
    let trail = matches!(mode, PathMode::Trail);
    let node_unique = matches!(mode, PathMode::Simple | PathMode::Acyclic);

    let mut keep: Vec<usize> = Vec::new();
    let mut ends: Vec<u32> = Vec::new();
    let mut cols: Vec<Vec<Value>> = vec![Vec::new(); bind_slots.len()];

    // Recursion state, carried in a small struct to keep the many closures honest.
    struct M<'a> {
        store: &'a Store,
        unit: &'a crate::ir::GUnit,
        per_rep: Option<&'a Expr>,
        omin: u32,
        omax: u32,
        trail: bool,
        node_unique: bool,
        used_edges: Vec<u32>,
        used_nodes: Vec<u32>,
        steps: Vec<StepRec>,
    }
    let mut m = M {
        store,
        unit,
        per_rep: per_rep_pred,
        omin: min,
        omax: max,
        trail,
        node_unique,
        used_edges: Vec::new(),
        used_nodes: Vec::new(),
        steps: Vec::new(),
    };

    impl M<'_> {
        // One hop from `v` (edge types `want`, direction `dir`, per-hop `epred`), tagged
        // with `levels`. Calls `f(target)` per admissible neighbour, StepRec pushed;
        // restores on return.
        fn do_hop(
            &mut self,
            v: u32,
            want: &[u32],
            dir: Dir,
            epred: Option<&Expr>,
            levels: Vec<(u32, usize)>,
            f: &mut dyn FnMut(&mut Self, u32),
        ) {
            let mut adjs: Vec<crate::store::Adj> = Vec::new();
            if matches!(dir, Dir::Out | Dir::Both) {
                adjs.extend_from_slice(self.store.out(v));
            }
            if matches!(dir, Dir::In | Dir::Both) {
                adjs.extend_from_slice(self.store.inc(v));
            }
            for a in adjs {
                if !edge_carries_wanted(self.store, &a, want) {
                    continue;
                }
                if !edge_pred_ok(epred, self.store, a.eid) {
                    continue; // per-hop edge WHERE / inline props
                }
                if self.trail && self.used_edges.contains(&a.eid) {
                    continue;
                }
                if self.node_unique && self.used_nodes.contains(&a.nbr) {
                    continue;
                }
                self.steps.push(StepRec {
                    levels: levels.clone(),
                    source: v,
                    edge: a.eid,
                    target: a.nbr,
                });
                if self.trail {
                    self.used_edges.push(a.eid);
                }
                if self.node_unique {
                    self.used_nodes.push(a.nbr);
                }
                f(self, a.nbr);
                if self.node_unique {
                    self.used_nodes.pop();
                }
                if self.trail {
                    self.used_edges.pop();
                }
                self.steps.pop();
            }
        }

        // Match the OUTER unit's element sequence `outer.elems[ei..]` from `v`, then
        // `cont(end)`. A direct hop advances one element (levels `[(orep, ei+1)]`); a
        // Sub repeats its flat inner unit before continuing (levels
        // `[(orep, ei), (irep, ihop+1)]` for its inner hops).
        fn seq(&mut self, v: u32, ei: usize, orep: u32, cont: &mut dyn FnMut(&mut Self, u32)) {
            let outer = self.unit; // copy the &GUnit so `self` stays free for the calls
            if ei == outer.elems.len() {
                cont(self, v);
                return;
            }
            match &outer.elems[ei] {
                GElem::Hop {
                    dir,
                    etypes,
                    edge_pred,
                    ..
                } => {
                    let want = want_etypes(self.store, etypes).unwrap_or_else(|()| vec![u32::MAX]);
                    let (dir, epred) = (*dir, edge_pred.as_deref());
                    self.do_hop(
                        v,
                        &want,
                        dir,
                        epred,
                        vec![(orep, ei + 1)],
                        &mut |slf, nbr| slf.seq(nbr, ei + 1, orep, cont),
                    );
                }
                GElem::Sub {
                    unit: sub,
                    min,
                    max,
                    ..
                } => {
                    let (smin, smax) = (*min, *max);
                    self.sub_walk(v, sub, smin, smax, orep, ei, 0, &mut |slf, end| {
                        slf.seq(end, ei + 1, orep, cont)
                    });
                }
            }
        }

        // Repeat a Sub's flat inner unit [smin,smax] times from `v`; `cont(end)` at each
        // inner-rep-count boundary in range.
        #[allow(clippy::too_many_arguments)]
        fn sub_walk(
            &mut self,
            v: u32,
            sub: &crate::ir::GUnit,
            smin: u32,
            smax: u32,
            orep: u32,
            es: usize,
            irep: u32,
            cont: &mut dyn FnMut(&mut Self, u32),
        ) {
            if irep >= smin {
                cont(self, v);
            }
            if irep < smax {
                self.sub_rep(v, sub, 0, orep, es, irep, &mut |slf, end| {
                    slf.sub_walk(end, sub, smin, smax, orep, es, irep + 1, cont)
                });
            }
        }

        // Match one inner rep (the Sub's flat hops) from `v`, then `cont(end)`.
        #[allow(clippy::too_many_arguments)]
        fn sub_rep(
            &mut self,
            v: u32,
            sub: &crate::ir::GUnit,
            ihop: usize,
            orep: u32,
            es: usize,
            irep: u32,
            cont: &mut dyn FnMut(&mut Self, u32),
        ) {
            if ihop == sub.elems.len() {
                cont(self, v);
                return;
            }
            let GElem::Hop {
                dir,
                etypes,
                edge_pred,
                ..
            } = &sub.elems[ihop]
            else {
                return;
            };
            let want = want_etypes(self.store, etypes).unwrap_or_else(|()| vec![u32::MAX]);
            let (dir, epred) = (*dir, edge_pred.as_deref());
            self.do_hop(
                v,
                &want,
                dir,
                epred,
                vec![(orep, es), (irep, ihop + 1)],
                &mut |slf, nbr| slf.sub_rep(nbr, sub, ihop + 1, orep, es, irep, cont),
            );
        }

        // The PER-REP `WHERE` over the just-completed outer rep `orep`: bind the unit's
        // variables in the per-rep view (`key_start = 1`, over that rep's steps) and
        // evaluate. `true` when there is no predicate. A rep failing it is pruned.
        fn rep_ok(&self, orep: u32) -> bool {
            let Some(pred) = self.per_rep else {
                return true;
            };
            let rep_steps: Vec<StepRec> = self
                .steps
                .iter()
                .filter(|s| s.levels.first().is_some_and(|(r, _)| *r == orep))
                .cloned()
                .collect();
            let mut pairs: Vec<(usize, Value)> = Vec::new();
            bind_nested(self.unit, &[], 1, &rep_steps, &mut pairs);
            let maxslot = pairs.iter().map(|(s, _)| *s).max().unwrap_or(0);
            let mut cols: Vec<Col> = (0..=maxslot).map(|_| Col::Gen(vec![Value::Null])).collect();
            for (s, v) in pairs {
                cols[s] = Col::Gen(vec![v]);
            }
            let mini = Batch::of(cols);
            eval(pred, self.store, &mini)
                .map(|c| c.value_at(0).is_true())
                .unwrap_or(false)
        }

        // The outer repetition: repeat the whole unit [omin,omax] times from `v`,
        // emitting the endpoint at each outer-rep-count boundary in range. A completed
        // outer rep that fails the per-rep `WHERE` prunes that branch.
        fn outer_walk(&mut self, v: u32, orep: u32, emit: &mut dyn FnMut(&mut Self, u32)) {
            if orep >= self.omin {
                emit(self, v);
            }
            if orep < self.omax {
                let mut c = |slf: &mut Self, end: u32| {
                    if slf.rep_ok(orep) {
                        slf.outer_walk(end, orep + 1, emit);
                    }
                };
                self.seq(v, 0, orep, &mut c);
            }
        }
    }

    for (row, &s) in src.iter().enumerate() {
        if m.node_unique {
            m.used_nodes.push(s);
        }
        let mut emit = |slf: &mut M, end: u32| {
            keep.push(row);
            ends.push(end);
            let mut pairs: Vec<(usize, Value)> = Vec::new();
            bind_nested(unit, &[], 0, &slf.steps, &mut pairs);
            for (ci, &want_slot) in bind_slots.iter().enumerate() {
                let v = pairs
                    .iter()
                    .find(|(sl, _)| *sl == want_slot)
                    .map(|(_, v)| v.clone())
                    .unwrap_or(Value::Null);
                cols[ci].push(v);
            }
        };
        m.outer_walk(s, 0, &mut emit);
        if m.node_unique {
            m.used_nodes.pop();
        }
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(ends));
    for c in cols {
        slots.push(Col::Gen(c));
    }
    Batch::of(slots)
}

/// Depth-first path enumeration for `var_length`. Emits `(row, endpoint)` at
/// every length in `min..=max` reached from the source, pushing straight into
/// `keep`/`ends` (a recursion-friendly alternative to a closure). `used` holds the
/// on-path elements that block reuse under `mode` — edge ids for a trail, node ids
/// for Simple/Acyclic (seeded with `start`). See [`varlen_step`].
#[allow(clippy::too_many_arguments)]
/// The per-path emit action of the var-length DFS. Parameterizing it lets the same
/// traversal either MATERIALIZE rows into a batch ([`CollectEmit`]) or STREAM them
/// straight to an output sink without ever building the batch — the memory win for a
/// huge closure. Each emit is a completed path whose endpoint is `node_stack.last()`.
trait VarlenEmit {
    fn emit(&mut self, row: usize, node_stack: &[u32], edge_stack: &[u32]);
    /// Stop descending — the emit budget (or a stream's output cap) is exhausted.
    fn should_stop(&self) -> bool;
}

/// The materializing emit: reproduces exactly the old inline `keep`/`ends`/lineage/group
/// pushes, so a batch built through it is byte-identical to before the refactor.
struct CollectEmit<'a> {
    keep: Vec<usize>,
    ends: Vec<u32>,
    bufs: PathBufs,
    group_cols: Vec<Vec<Value>>,
    batch: Option<&'a Batch>,
    group_binds: &'a [(crate::ir::GroupPos, usize)],
    k: u32,
    budget: u64,
}

impl VarlenEmit for CollectEmit<'_> {
    fn emit(&mut self, row: usize, node_stack: &[u32], edge_stack: &[u32]) {
        let endpoint = *node_stack.last().expect("a path always has an endpoint");
        self.keep.push(row);
        self.ends.push(endpoint);
        if let Some(b) = self.batch {
            push_path(
                b,
                row,
                node_stack,
                edge_stack,
                &mut self.bufs.values,
                &mut self.bufs.offsets,
                &mut self.bufs.edges,
                &mut self.bufs.edge_offsets,
            );
        }
        if !self.group_binds.is_empty() {
            push_group_cols(
                node_stack,
                edge_stack,
                self.k,
                self.group_binds,
                &mut self.group_cols,
            );
        }
    }
    fn should_stop(&self) -> bool {
        self.keep.len() as u64 > self.budget
    }
}

/// A var-length sink that keeps only the DISTINCT reachable endpoints, in first-seen order.
/// For `DISTINCT`/`min`/`max` over a var-length endpoint the answer depends only on the SET
/// of reachable endpoints (not per-path multiplicity), so emitting each endpoint once —
/// same DFS exploration — is byte-identical to materializing every path and then deduping,
/// while never building the (potentially millions of) path rows. It also has no emit
/// budget, so it COMPLETES the shapes the materializing walk refuses with E_RESOURCE
/// (matching core, which completes them).
struct DistinctEndpointEmit {
    seen: Vec<bool>,
    out: Vec<u32>,
}

impl VarlenEmit for DistinctEndpointEmit {
    fn emit(&mut self, _row: usize, node_stack: &[u32], _edge_stack: &[u32]) {
        let ep = *node_stack.last().expect("a path always has an endpoint");
        if !std::mem::replace(&mut self.seen[ep as usize], true) {
            self.out.push(ep);
        }
    }
    fn should_stop(&self) -> bool {
        false // explore fully; only distinct endpoints are kept, so memory stays bounded
    }
}

/// Drive the var-length DFS from every source in `src`, feeding each completed path to
/// `sink`. The per-source setup (Simple/Acyclic node marking, the seeded stacks) that used
/// to live inline in `var_length` — factored out so a streaming caller reuses the exact
/// same traversal.
#[allow(clippy::too_many_arguments)]
fn run_varlen<S: VarlenEmit>(
    src: &[u32],
    store: &Store,
    want: &[u32],
    min: u32,
    max: u32,
    dir: Dir,
    mode: PathMode,
    per_rep_pred: Option<&Expr>,
    k: u32,
    until_stop: Option<(&Expr, usize)>,
    body_filter: Option<(&Expr, usize)>,
    double_loops: bool,
    sink: &mut S,
) {
    let node_unique = matches!(mode, PathMode::Simple | PathMode::Acyclic);
    let mut used: Vec<u32> = Vec::new();
    for (row, &v) in src.iter().enumerate() {
        if node_unique {
            used.push(v);
        }
        let mut node_stack = vec![v];
        let mut edge_stack: Vec<u32> = Vec::new();
        varlen_walk(
            store,
            v,
            min,
            max,
            dir,
            want,
            mode,
            v,
            &mut used,
            row,
            &mut node_stack,
            &mut edge_stack,
            per_rep_pred,
            k,
            until_stop,
            body_filter,
            double_loops,
            sink,
        );
        if node_unique {
            used.pop();
        }
        debug_assert!(used.is_empty());
        if sink.should_stop() {
            break; // budget/cap tripped — do not start the next source's walk
        }
    }
}

/// The streaming emit: writes each emitted endpoint's property straight into a JSON array
/// buffer, never building a batch — so a huge var-length closure costs O(output text), not
/// O(rows × boxed Value). Bounded by a byte cap: past it we stop and the caller returns
/// `E_RESOURCE_EXHAUSTED` (a loud failure, not a silent truncation or an OOM). The bytes
/// written are identical to `read_property(endpoint) -> write_value`, so streamed output
/// equals the materialized `gremlin_results_json`.
struct StreamPropEmit<'a> {
    out: String,
    first: bool,
    // The property COLUMN, resolved once — a typed cell read per endpoint (no per-row
    // HashMap-by-key `store.prop`, which made the stream 1.3-1.6x slower than the
    // vectorized materialized read at medium sizes). `None` = the key has no column.
    col: Option<&'a Column>,
    // `values(k)` inserts a `PropertyExists{k}` filter above the hop: an endpoint MISSING
    // the property is dropped, not emitted as null. Mirror that skip when set. (A GQL
    // `RETURN t.k` has no such filter and emits null for an absent value.)
    require_present: bool,
    // GQL rows wrap each value in a 1-column array (`[v]`); a Gremlin value stream does not.
    wrap_row: bool,
    byte_cap: usize,
}

impl StreamPropEmit<'_> {
    fn sep(&mut self) {
        if !self.first {
            self.out.push(',');
        }
        self.first = false;
    }
}

impl VarlenEmit for StreamPropEmit<'_> {
    fn emit(&mut self, _row: usize, node_stack: &[u32], _edge_stack: &[u32]) {
        if self.out.len() > self.byte_cap {
            return; // over the cap — stop appending (should_stop halts descent)
        }
        let i = *node_stack.last().expect("a path always has an endpoint") as usize;
        let present = self.col.is_some_and(|c| c.present_at(i));
        if !present && self.require_present {
            return; // dropped by the PropertyExists guard
        }
        self.sep();
        if self.wrap_row {
            self.out.push('[');
        }
        if !present {
            self.out.push_str("null");
        } else {
            // Typed cell write — byte-identical to `read_property(endpoint) -> write_value`.
            match self.col.expect("present implies a column") {
                Column::Str { data, .. } => crate::json::write_string(&mut self.out, &data[i]),
                Column::Dict { dict, codes, .. } => {
                    crate::json::write_string(&mut self.out, &dict[codes[i] as usize]);
                }
                Column::Num { data, .. } => {
                    crate::json::write_value(&mut self.out, &Value::Num(data[i]));
                }
                Column::Bool { data, .. } => {
                    self.out.push_str(if data[i] { "true" } else { "false" });
                }
                other => crate::json::write_value(&mut self.out, &other.read(i)), // Temporal/Gen
            }
        }
        if self.wrap_row {
            self.out.push(']');
        }
    }
    fn should_stop(&self) -> bool {
        self.out.len() > self.byte_cap
    }
}

/// `g.V()…repeat(hop).values(k)` / GQL `RETURN t.k` over a plain var-length: stream the
/// endpoint property to a JSON array without materializing the (possibly huge) row batch.
/// Returns `None` when the shape isn't a single-`Prop` projection of the endpoint over a
/// plain `VarLength` (no lineage, no group binds); `Some(Err)` when the output byte cap is
/// hit; `Some(Ok(json))` otherwise. Byte-identical to serializing the materialized result.
fn try_stream_varlen_json(plan: &Plan, store: &Store, gql: bool) -> Option<Result<String, String>> {
    // Output cap — generous (compact text), so far more rows complete than the 1M-row
    // trail cap the materialized path hits, yet a runaway still fails loudly.
    const BYTE_CAP: usize = 256 << 20; // 256 MiB
    if needs_lineage(plan) {
        return None;
    }
    let Plan::Project { input, items } = plan else {
        return None;
    };
    let [(colname, Expr::Prop { slot, key })] = items.as_slice() else {
        return None;
    };
    // `values(k)` puts a `PropertyExists{k}` filter between the projection and the hop;
    // peel it (and honor its drop-if-absent semantics). Any other filter → fall back.
    let (vl, require_present) = match input.as_ref() {
        vl @ Plan::VarLength { .. } => (vl, false),
        Plan::Filter {
            input: fin,
            pred: Expr::PropertyExists { slot: fs, key: fk },
        } if fs == slot && fk == key => (fin.as_ref(), true),
        _ => return None,
    };
    let Plan::VarLength {
        input: vl_in,
        from,
        dir,
        edge_label,
        min,
        max,
        mode,
        until,
        body_filter,
        double_loops,
    } = vl
    else {
        return None;
    };
    let vl_batch = pull(vl_in, store, false).ok()?;
    let endpoint_slot = vl_batch.slots.len();
    if *slot != endpoint_slot {
        return None; // the projection reads some bound var other than the endpoint
    }
    let Col::Nodes(src) = vl_batch.slot(*from) else {
        return None;
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => vec![u32::MAX],
    };
    // Envelope: Gremlin is a bare value array `[…]`; GQL is `{columns:[name],rows:[[v],…]}`.
    let out = if gql {
        let mut p = String::from("{\"columns\":[");
        crate::json::write_string(&mut p, colname);
        p.push_str("],\"rows\":[");
        p
    } else {
        String::from("[")
    };
    let mut sink = StreamPropEmit {
        out,
        first: true,
        col: store.column(key),
        require_present,
        wrap_row: gql,
        byte_cap: BYTE_CAP,
    };
    run_varlen(
        src,
        store,
        &want,
        *min,
        *max,
        *dir,
        *mode,
        None, // a plain VarLength has no per-rep predicate (that is RepeatGroup)
        1,
        until.as_deref().map(|p| (p, endpoint_slot)),
        body_filter.as_deref().map(|p| (p, endpoint_slot)),
        *double_loops,
        &mut sink,
    );
    if sink.out.len() > BYTE_CAP {
        return Some(Err(format!(
            "E_RESOURCE_EXHAUSTED: variable-length traversal output exceeded {BYTE_CAP} bytes; \
             add a tighter bound/`LIMIT` or dedup the frontier"
        )));
    }
    sink.out.push_str(if gql { "]}" } else { "]" });
    Some(Ok(sink.out))
}

/// The pre-work verdict for a node reached at depth `len`: does the walk emit here,
/// and should it descend into this node's adjacency? Mirrors the early-return ladder
/// at the top of the recursive body so the iterative driver stays byte-identical.
enum Enter {
    /// Budget/cap tripped — abort the whole walk (unwind, no more emits).
    Stop,
    /// This node contributes no descent (a filter pruned it, an `until` matched, or
    /// `len == max`). Any emit for it has already fired.
    Prune,
    /// Emit (if at a boundary) done; iterate this node's adjacency.
    Iterate,
}

/// Node-entry pre-work: `should_stop`, body filter, per-rep predicate, the boundary
/// emit, the `until`-match prune, and the `len == max` stop — in the SAME order and
/// with the SAME emit point as the recursive `varlen_dfs`. Split out so the iterative
/// driver runs it once per node exactly as the recursion did.
#[allow(clippy::too_many_arguments)]
fn varlen_enter<S: VarlenEmit>(
    store: &Store,
    v: u32,
    len: u32,
    min: u32,
    max: u32,
    k: u32,
    per_rep_pred: Option<&Expr>,
    until_stop: Option<(&Expr, usize)>,
    body_filter: Option<(&Expr, usize)>,
    node_stack: &[u32],
    edge_stack: &[u32],
    row: usize,
    sink: &mut S,
) -> Enter {
    if sink.should_stop() {
        return Enter::Stop;
    }
    if len > 0 {
        if let Some((pred, slot)) = body_filter {
            if !until_ok(pred, store, slot, v) {
                return Enter::Prune;
            }
        }
    }
    if let Some(pred) = per_rep_pred {
        if len > 0
            && len.is_multiple_of(k)
            && !rep_pred_ok(pred, store, node_stack, edge_stack, len, k)
        {
            return Enter::Prune;
        }
    }
    let at_boundary = len >= min && len.is_multiple_of(k);
    let until_hit = until_stop.map(|(p, slot)| until_ok(p, store, slot, v));
    let emit_here = at_boundary && until_hit.unwrap_or(true);
    if emit_here {
        sink.emit(row, node_stack, edge_stack);
    }
    if until_stop.is_some() && emit_here {
        return Enter::Prune;
    }
    if len == max {
        return Enter::Prune;
    }
    Enter::Iterate
}

/// The `i`th adjacency entry of `v` under `dir`, in the recursion's order: the whole
/// OUT slice first, then the whole IN slice. Recomputes the slices per call (each is an
/// O(1) borrow), so the driver need not hold a borrow of `store` across the frame.
#[inline]
fn adj_nth(store: &Store, v: u32, dir: Dir, i: usize) -> Option<(bool, crate::store::Adj)> {
    let out: &[crate::store::Adj] = if matches!(dir, Dir::Out | Dir::Both) {
        store.out(v)
    } else {
        &[]
    };
    if i < out.len() {
        return Some((false, out[i]));
    }
    let inc: &[crate::store::Adj] = if matches!(dir, Dir::In | Dir::Both) {
        store.inc(v)
    } else {
        &[]
    };
    inc.get(i - out.len()).map(|a| (true, *a))
}

/// One open node on the walk's path: where we are (`v`, `len`), how far through its
/// adjacency we've iterated (`cursor`), and — once we descend — how to undo the
/// child's push when we resume (`pending`: `Some(true)` also pops a `used` mark).
struct VarFrame {
    v: u32,
    len: u32,
    cursor: usize,
    pending: Option<bool>,
}

/// Iterative equivalent of [`varlen_dfs`] — an explicit heap-allocated frame stack in
/// place of call recursion, so a deep closure costs heap (bounded, cheap to grow), not
/// call-stack pages. This removes the multi-hundred-MB stack a deep path used to commit
/// (and the 1 GiB scoped thread that hosted it): peak memory is now the frame stack plus
/// the path stacks, all `O(current depth)`.
///
/// Byte-identical to the recursion for every walk that stays under the budget/cap: same
/// pre-order emit points (via [`varlen_enter`]), same OUT-then-IN adjacency order (via
/// [`adj_nth`]), same `Close` handling. Past the cap both return `E_RESOURCE` and their
/// partial output is discarded, so the exact over-cap emit count is unobservable — the
/// driver just tears down promptly, restoring `used`/the path stacks for the assert in
/// [`run_varlen`].
#[allow(clippy::too_many_arguments)]
fn varlen_walk<S: VarlenEmit>(
    store: &Store,
    v0: u32,
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    mode: PathMode,
    start: u32,
    used: &mut Vec<u32>,
    row: usize,
    node_stack: &mut Vec<u32>,
    edge_stack: &mut Vec<u32>,
    per_rep_pred: Option<&Expr>,
    k: u32,
    until_stop: Option<(&Expr, usize)>,
    body_filter: Option<(&Expr, usize)>,
    double_loops: bool,
    sink: &mut S,
) {
    // The source's own pre-work (an emit at `len == 0` for `min == 0`, filters). If it
    // does not descend, there is nothing to walk.
    match varlen_enter(
        store,
        v0,
        0,
        min,
        max,
        k,
        per_rep_pred,
        until_stop,
        body_filter,
        node_stack,
        edge_stack,
        row,
        sink,
    ) {
        Enter::Iterate => {}
        Enter::Prune | Enter::Stop => return,
    }
    let drop_loop = matches!(dir, Dir::Both) && !double_loops;
    let mut stack: Vec<VarFrame> = vec![VarFrame {
        v: v0,
        len: 0,
        cursor: 0,
        pending: None,
    }];
    // Tear the whole stack down, undoing each frame's pending child push — keeps `used`
    // and the path stacks clean on an abort so the next source starts fresh.
    macro_rules! teardown {
        () => {{
            while let Some(f) = stack.pop() {
                if let Some(pop_used) = f.pending {
                    node_stack.pop();
                    edge_stack.pop();
                    if pop_used {
                        used.pop();
                    }
                }
            }
            return;
        }};
    }
    'frames: while let Some(top) = stack.last_mut() {
        // Resuming after a child finished: undo the push that descent made.
        if let Some(pop_used) = top.pending.take() {
            node_stack.pop();
            edge_stack.pop();
            if pop_used {
                used.pop();
            }
        }
        let (v, len) = (top.v, top.len);
        loop {
            let cursor = top.cursor;
            let Some((is_inc, a)) = adj_nth(store, v, dir, cursor) else {
                stack.pop(); // this node's adjacency is exhausted
                continue 'frames;
            };
            top.cursor += 1;
            // Edge must carry a wanted label (primary type or, multi-label, a secondary).
            if !want.is_empty()
                && !want.iter().any(|&w| {
                    w == a.etype
                        || (store.has_multi_label_edges() && store.edge_has_label(a.eid, w))
                })
            {
                continue;
            }
            if is_inc && drop_loop && a.nbr == v {
                continue;
            }
            let mark = match varlen_step(mode, start, &a, used) {
                VarStep::Skip => continue,
                VarStep::Close => {
                    // Closing hop (Simple cycle back to `start`): emit at a rep boundary,
                    // never descend. Push/emit/pop so the path is complete for the sink.
                    if len + 1 >= min && (len + 1).is_multiple_of(k) {
                        node_stack.push(a.nbr);
                        edge_stack.push(a.eid);
                        sink.emit(row, node_stack, edge_stack);
                        node_stack.pop();
                        edge_stack.pop();
                    }
                    continue;
                }
                VarStep::Go(mark) => mark,
            };
            if let Some(m) = mark {
                used.push(m);
            }
            node_stack.push(a.nbr);
            edge_stack.push(a.eid);
            match varlen_enter(
                store,
                a.nbr,
                len + 1,
                min,
                max,
                k,
                per_rep_pred,
                until_stop,
                body_filter,
                node_stack,
                edge_stack,
                row,
                sink,
            ) {
                Enter::Iterate => {
                    // Descend: remember how to undo this push, push the child frame.
                    top.pending = Some(mark.is_some());
                    stack.push(VarFrame {
                        v: a.nbr,
                        len: len + 1,
                        cursor: 0,
                        pending: None,
                    });
                    continue 'frames;
                }
                Enter::Prune => {
                    // No descent — undo the push now and try the next sibling edge.
                    node_stack.pop();
                    edge_stack.pop();
                    if mark.is_some() {
                        used.pop();
                    }
                    continue;
                }
                Enter::Stop => {
                    node_stack.pop();
                    edge_stack.pop();
                    if mark.is_some() {
                        used.pop();
                    }
                    teardown!();
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn varlen_dfs<S: VarlenEmit>(
    store: &Store,
    v: u32,
    len: u32,
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    mode: PathMode,
    start: u32,
    used: &mut Vec<u32>,
    row: usize,
    // `node_stack`/`edge_stack` hold the chain source..`v`; the emit reads the endpoint
    // (`node_stack.last()`) and the whole path from them.
    node_stack: &mut Vec<u32>,
    edge_stack: &mut Vec<u32>,
    // A per-repetition predicate over the rep's scalar (source=0, edge=1, target=2);
    // a hop that fails it is not descended into (the path through it is pruned).
    per_rep_pred: Option<&Expr>,
    // Hops per repetition unit: an endpoint is emitted only at a rep boundary.
    k: u32,
    // Gremlin `until(pred)`: `(pred, endpoint_slot)`. An endpoint is emitted ONLY when
    // `pred` holds at it, and the branch then PRUNES (no descent past the match).
    until_stop: Option<(&Expr, usize)>,
    // Gremlin `repeat(<hop>.<filter>)` body filter: `(pred, endpoint_slot)`. Applied at
    // each hop target (`len > 0`); a target that fails it is pruned entirely.
    body_filter: Option<(&Expr, usize)>,
    // Gremlin `both()` self-loop doubling — keep the in-side copy of a self-loop.
    double_loops: bool,
    // Where each completed path goes (materialize or stream), and the stop condition.
    sink: &mut S,
) {
    // Budget/cap tripped by a sibling branch — stop descending (the walk is aborting).
    if sink.should_stop() {
        return;
    }
    // A body filter prunes a hop target that fails it — no emit, no descent (the source
    // at len 0 is not a hop target, so it is exempt).
    if len > 0 {
        if let Some((pred, slot)) = body_filter {
            if !until_ok(pred, store, slot, v) {
                return;
            }
        }
    }
    // Per-repetition WHERE: on COMPLETING a rep (a boundary at len > 0), check the
    // just-finished rep's predicate over its scalar variables (node pos p at slot 2p,
    // edge pos p at 2p+1). A per-rep WHERE must hold for EVERY rep, so a failing rep
    // invalidates the whole path onward — prune (no emit, no descent).
    if let Some(pred) = per_rep_pred {
        if len > 0
            && len.is_multiple_of(k)
            && !rep_pred_ok(pred, store, node_stack, edge_stack, len, k)
        {
            return;
        }
    }
    // With `until(pred)` a landing is emitted ONLY when the predicate holds; a plain
    // walk emits every landing in `[min, max]`. On an `until` match the branch prunes.
    let at_boundary = len >= min && len.is_multiple_of(k);
    let until_hit = until_stop.map(|(p, slot)| until_ok(p, store, slot, v));
    let emit_here = at_boundary && until_hit.unwrap_or(true);
    if emit_here {
        // `node_stack` ends at `v` (the recursion pushed it), so it IS this path.
        sink.emit(row, node_stack, edge_stack);
    }
    // An `until` match at an emit boundary stops the walk here (the loop exit); a match
    // BELOW `min` (e.g. a post-form do-while source that already satisfies `pred`) does
    // NOT — the body must still run its minimum iterations first. So prune on `emit_here`.
    if until_stop.is_some() && emit_here {
        return;
    }
    if len == max {
        return;
    }
    // Iterate the OUT then IN adjacency slices directly — chained, not copied into
    // a per-visit `Vec` (that allocation, once per node on a path of which there are
    // millions, dominated). Order is unchanged (out first, then in), so the emitted
    // path multiset and its order are bit-identical.
    let out: &[crate::store::Adj] = if matches!(dir, Dir::Out | Dir::Both) {
        store.out(v)
    } else {
        &[]
    };
    let inc: &[crate::store::Adj] = if matches!(dir, Dir::In | Dir::Both) {
        store.inc(v)
    } else {
        &[]
    };
    // Undirected: drop the in-side copy of a self-loop so it is walked once.
    let drop_loop = matches!(dir, Dir::Both) && !double_loops;
    for (is_inc, a) in out
        .iter()
        .map(|a| (false, a))
        .chain(inc.iter().map(|a| (true, a)))
    {
        // The edge must carry a wanted label — its primary type (`a.etype`) or, on
        // a multi-label graph, a secondary one (`edge_has_label`).
        if !want.is_empty()
            && !want.iter().any(|&w| {
                w == a.etype || (store.has_multi_label_edges() && store.edge_has_label(a.eid, w))
            })
        {
            continue;
        }
        if is_inc && drop_loop && a.nbr == v {
            continue;
        }
        // (A per-repetition WHERE is checked at the rep boundary on the way IN — see
        // the top of this function — not per hop, so a multi-hop rep sees all its
        // edges bound.)
        let mark = match varlen_step(mode, start, a, used) {
            VarStep::Skip => continue,
            VarStep::Close => {
                // Emit the closing endpoint (the start) at this length, no descent —
                // only at a rep boundary for a multi-hop unit. Push the closing node/edge
                // so the path (and its `node_stack.last()` endpoint) is complete, then pop.
                if len + 1 >= min && (len + 1).is_multiple_of(k) {
                    node_stack.push(a.nbr);
                    edge_stack.push(a.eid);
                    sink.emit(row, node_stack, edge_stack);
                    node_stack.pop();
                    edge_stack.pop();
                }
                continue;
            }
            VarStep::Go(mark) => mark,
        };
        if let Some(m) = mark {
            used.push(m);
        }
        node_stack.push(a.nbr);
        edge_stack.push(a.eid);
        varlen_dfs(
            store,
            a.nbr,
            len + 1,
            min,
            max,
            dir,
            want,
            mode,
            start,
            used,
            row,
            node_stack,
            edge_stack,
            per_rep_pred,
            k,
            until_stop,
            body_filter,
            double_loops,
            sink,
        );
        node_stack.pop();
        edge_stack.pop();
        if mark.is_some() {
            used.pop();
        }
    }
}

/// Shortest-path reach: a BFS from each input row's source node, emitting each
/// reachable target ONCE at its shortest distance (the first BFS reach), with the
/// target appended as a new slot. ANY-shortest — one representative per target,
/// not every shortest path. The source is not emitted; `max` caps hop distance.
#[allow(clippy::too_many_arguments)]
fn shortest_path(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: &[String],
    min: u32,
    max: Option<u32>,
    selector: crate::ir::ShortestSelector,
    edge_pred: Option<&Expr>,
    // Target-aware early stop (the reverse-walk win): when the endpoint is constrained to
    // a resolved node set (a `Filter{endpoint == t}` fused in above), bound each source's
    // BFS to the deepest target's distance instead of sweeping the whole component. Pure
    // optimization — the outer filter still runs, so the KEPT rows (targets, with their
    // shortest paths and multiplicity) are byte-identical; only never-kept nodes beyond
    // the targets go unexplored. Applied only for `min == 0` (no `+`-cycle source cases).
    early_stop: Option<&[u32]>,
) -> Batch {
    use crate::ir::ShortestSelector;
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        Batch::of(slots)
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        // An unknown edge type matches NO edge — but the zero-length path traverses
        // none, so a `*` (min == 0) still emits each source at length 0. Fall through
        // with a never-matching set (NOT the empty "any" set) so the BFS traverses
        // nothing yet the emit loop still yields the length-0 sources.
        Err(()) => vec![u32::MAX],
    };
    // `SHORTEST k` (k >= 2) needs paths BEYOND the single shortest length, so it
    // can't ride the BFS below — enumerate trails and select per endpoint instead.
    if let ShortestSelector::ShortestK { k, group } = selector {
        return shortest_k_path(
            batch, store, from, dir, &want, min, max, k, group, edge_pred,
        );
    }
    let Col::Nodes(src) = batch.slot(from) else {
        return empty();
    };

    // Target membership bitset for the early stop (only `min == 0`, so a target
    // discovered at any depth is a valid endpoint — no `+`-cycle subtlety). `remaining`
    // per source counts distinct targets still undiscovered; when it hits 0 the last
    // target's distance `D` is the deepest, and every target's shortest predecessors sit
    // at depth < D, so once we pop a node at depth >= D all targets are fully settled.
    let n_nodes = store.node_count();
    let target_bits: Option<Vec<bool>> = early_stop.filter(|_| min == 0).map(|ts| {
        let mut b = vec![false; n_nodes];
        for &t in ts {
            if (t as usize) < n_nodes {
                b[t as usize] = true;
            }
        }
        b
    });
    let remaining_init: usize = target_bits
        .as_ref()
        .map_or(0, |b| b.iter().filter(|&&x| x).count());

    // When the input carries a path, each emitted row reconstructs the node/edge
    // chain start..endpoint so `Expr::Path` (nodes(p)/path_length(p)) sees the whole
    // path. Endpoint MULTIPLICITY (how many shortest paths reach it) is emitted for
    // `ALL` regardless of lineage.
    let track = batch.lineage.is_some();
    let mut path_values: Vec<Value> = Vec::new();
    let mut path_offsets: Vec<usize> = vec![0];
    let mut path_edges: Vec<Value> = Vec::new();
    let mut path_edge_offsets: Vec<usize> = vec![0];

    let mut keep = Vec::new();
    let mut ends = Vec::new();

    for (row, &start) in src.iter().enumerate() {
        // BFS from `start`: shortest distance per node, plus ALL predecessors that
        // lie on a shortest path (an edge prev->node with dist[prev] + 1 == dist[node]).
        // `order` is BFS discovery order — every node's predecessors precede it, so a
        // bottom-up pass over it computes path counts / enumerations.
        let mut dist: FnvMap<u32, u32> = FnvMap::default();
        dist.insert(start, 0);
        let mut preds: FnvMap<u32, Vec<(u32, u32)>> = FnvMap::default();
        let mut order: Vec<u32> = vec![start];
        let mut q: VecDeque<u32> = VecDeque::new();
        q.push_back(start);
        // Edges that CLOSE a cycle back to `start` as `(len, tail, eid)` where `len =
        // dist[tail] + 1`. A `+`-style (min >= 1) quantifier treats the source as a
        // valid endpoint at the shortest such length (a cycle) — standard BFS never
        // re-reaches `start`, so collect the closing edges here.
        let mut cycle_edges: Vec<(u32, u32, u32)> = Vec::new();
        // Early-stop bookkeeping (a no-op when `target_bits` is None).
        let mut remaining = remaining_init;
        let mut stop_dist: Option<u32> = None;
        if let Some(bits) = &target_bits {
            if bits[start as usize] && remaining > 0 {
                remaining -= 1;
                if remaining == 0 {
                    stop_dist = Some(0); // source IS the only target — nothing to explore
                }
            }
        }
        while let Some(v) = q.pop_front() {
            let dv = dist[&v];
            // All targets settled and their predecessors (depth < D) fully dequeued: any
            // node at depth >= D is beyond every target and would be filtered out anyway.
            if stop_dist.is_some_and(|d| dv >= d) {
                break;
            }
            if max.is_some_and(|m| dv >= m) {
                continue; // hop cap: do not expand past `max`
            }
            let mut adjs: Vec<crate::store::Adj> = Vec::new();
            if matches!(dir, Dir::Out | Dir::Both) {
                adjs.extend_from_slice(store.out(v));
            }
            if matches!(dir, Dir::In | Dir::Both) {
                adjs.extend_from_slice(store.inc(v));
            }
            for a in adjs {
                if !edge_carries_wanted(store, &a, &want) || !edge_pred_ok(edge_pred, store, a.eid)
                {
                    continue;
                }
                if min >= 1 && a.nbr == start {
                    cycle_edges.push((dv + 1, v, a.eid));
                }
                match dist.get(&a.nbr).copied() {
                    None => {
                        dist.insert(a.nbr, dv + 1);
                        preds.entry(a.nbr).or_default().push((v, a.eid));
                        order.push(a.nbr);
                        q.push_back(a.nbr);
                        // First time we reach a target: once all are found, the deepest is
                        // at `dv + 1`, so stop after the current (depth `dv`) level drains.
                        if let Some(bits) = &target_bits {
                            if bits[a.nbr as usize] && remaining > 0 {
                                remaining -= 1;
                                if remaining == 0 {
                                    stop_dist = Some(dv + 1);
                                }
                            }
                        }
                    }
                    // Another edge onto a node at its shortest distance — a second
                    // shortest-path predecessor.
                    Some(dn) if dn == dv + 1 => {
                        preds.entry(a.nbr).or_default().push((v, a.eid));
                    }
                    _ => {}
                }
            }
        }

        // `ALL` without lineage: emit each endpoint as many times as it has distinct
        // shortest paths (count DP over `order`).
        let mut pcount: FnvMap<u32, u64> = FnvMap::default();
        if matches!(selector, ShortestSelector::All) && !track {
            pcount.insert(start, 1);
            for &node in &order {
                if node == start {
                    continue;
                }
                let c = preds
                    .get(&node)
                    .map(|ps| {
                        ps.iter()
                            .map(|&(p, _)| pcount.get(&p).copied().unwrap_or(0))
                            .sum()
                    })
                    .unwrap_or(0);
                pcount.insert(node, c);
            }
        }

        for &node in &order {
            let dn = dist[&node];
            if dn < min {
                continue; // a `+` quantifier (min 1) excludes the zero-length seed
            }
            match selector {
                // `ShortestK` returned early to `shortest_k_path`; only Any/All here.
                ShortestSelector::ShortestK { .. } => unreachable!("ShortestK routed away"),
                ShortestSelector::Any => {
                    keep.push(row);
                    ends.push(node);
                    if track {
                        let (chain, echain) = first_pred_chain(node, start, &preds);
                        push_path(
                            batch,
                            row,
                            &chain,
                            &echain,
                            &mut path_values,
                            &mut path_offsets,
                            &mut path_edges,
                            &mut path_edge_offsets,
                        );
                    }
                }
                ShortestSelector::All => {
                    if track {
                        for (chain, echain) in enumerate_shortest_paths(node, start, &preds) {
                            keep.push(row);
                            ends.push(node);
                            push_path(
                                batch,
                                row,
                                &chain,
                                &echain,
                                &mut path_values,
                                &mut path_offsets,
                                &mut path_edges,
                                &mut path_edge_offsets,
                            );
                        }
                    } else {
                        for _ in 0..pcount.get(&node).copied().unwrap_or(0) {
                            keep.push(row);
                            ends.push(node);
                        }
                    }
                }
            }
        }

        // A `+`-style (min >= 1) quantifier admits the SOURCE as an endpoint at the
        // shortest CYCLE length back to it — the length-0 self-path is excluded, so
        // `start` re-reached via a non-trivial path is its shortest match. The global
        // shortest cycle = min over closing edges `tail -> start` of `dist[tail] + 1`.
        if min >= 1 {
            if let Some(cyc) = cycle_edges.iter().map(|&(d, _, _)| d).min() {
                if cyc >= min && max.is_none_or(|m| cyc <= m) {
                    match selector {
                        ShortestSelector::ShortestK { .. } => unreachable!("ShortestK routed away"),
                        ShortestSelector::Any => {
                            keep.push(row);
                            ends.push(start);
                            if track {
                                let &(_, tail, eid) =
                                    cycle_edges.iter().find(|&&(d, _, _)| d == cyc).unwrap();
                                let (mut chain, mut echain) = first_pred_chain(tail, start, &preds);
                                chain.push(start);
                                echain.push(eid);
                                push_path(
                                    batch,
                                    row,
                                    &chain,
                                    &echain,
                                    &mut path_values,
                                    &mut path_offsets,
                                    &mut path_edges,
                                    &mut path_edge_offsets,
                                );
                            }
                        }
                        ShortestSelector::All => {
                            for &(_, tail, eid) in cycle_edges.iter().filter(|&&(d, _, _)| d == cyc)
                            {
                                if track {
                                    for (mut chain, mut echain) in
                                        enumerate_shortest_paths(tail, start, &preds)
                                    {
                                        chain.push(start);
                                        echain.push(eid);
                                        keep.push(row);
                                        ends.push(start);
                                        push_path(
                                            batch,
                                            row,
                                            &chain,
                                            &echain,
                                            &mut path_values,
                                            &mut path_offsets,
                                            &mut path_edges,
                                            &mut path_edge_offsets,
                                        );
                                    }
                                } else {
                                    for _ in 0..pcount.get(&tail).copied().unwrap_or(1) {
                                        keep.push(row);
                                        ends.push(start);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(ends));
    let mut out = Batch::of(slots);
    if track {
        out.lineage = Some(Lineage {
            values: path_values,
            offsets: path_offsets,
            edges: path_edges,
            edge_offsets: path_edge_offsets,
        });
    }
    out
}

/// A per-hop edge predicate (`-[e:R WHERE …]->`): TRUE (traverse) when there is no
/// predicate, else evaluate it over a one-row batch holding just the edge at slot 0.
/// A false / null / faulting predicate blocks the edge.
fn edge_pred_ok(pred: Option<&Expr>, store: &Store, eid: u32) -> bool {
    match pred {
        None => true,
        Some(p) => {
            let mini = Batch::of(vec![Col::Edges(vec![eid])]);
            eval(p, store, &mini)
                .map(|c| c.value_at(0).is_true())
                .unwrap_or(false)
        }
    }
}

/// Endpoint → its trails as `(length, node chain, edge chain)`, in discovery order.
type TrailsByEnd = FnvMap<u32, Vec<(u32, Vec<u32>, Vec<u32>)>>;

/// `SHORTEST k [GROUP]` (k >= 2): enumerate every TRAIL from each source (no edge
/// reuse → finite), group by endpoint, order each endpoint's trails by (length,
/// discovery), then keep the first `k` (plain) or every trail whose length is among
/// the `k` smallest distinct lengths (`group`). Mirrors core's `shortest_k_walk`;
/// the endpoint's own label/property filter is a `Filter` above this, so it selects
/// k per endpoint here and the filter narrows afterward.
#[allow(clippy::too_many_arguments)]
fn shortest_k_path(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    want: &[u32],
    min: u32,
    max: Option<u32>,
    k: u32,
    group: bool,
    edge_pred: Option<&Expr>,
) -> Batch {
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        Batch::of(slots)
    };
    let Col::Nodes(src) = batch.slot(from) else {
        return empty();
    };
    let track = batch.lineage.is_some();
    let cap = max.unwrap_or(u32::MAX);
    let mut keep = Vec::new();
    let mut ends = Vec::new();
    let mut bufs = PathBufs::new();
    for (row, &start) in src.iter().enumerate() {
        // endpoint -> its trails as (length, node chain, edge chain) in discovery
        // (DFS) order — the same order the stable length sort tie-breaks on.
        let mut per_end: TrailsByEnd = FnvMap::default();
        let mut node_stack = vec![start];
        let mut edge_stack: Vec<u32> = Vec::new();
        let mut used: Vec<u32> = Vec::new();
        collect_trails(
            store,
            start,
            0,
            min,
            cap,
            dir,
            want,
            edge_pred,
            &mut used,
            &mut node_stack,
            &mut edge_stack,
            &mut per_end,
        );
        let mut end_ids: Vec<u32> = per_end.keys().copied().collect();
        end_ids.sort_unstable();
        for end in end_ids {
            let mut paths = per_end.remove(&end).unwrap();
            paths.sort_by_key(|(len, _, _)| *len); // stable: discovery order within a length
            let selected: Vec<(Vec<u32>, Vec<u32>)> = if group {
                // Keep every trail at or below the k-th smallest DISTINCT length.
                let mut distinct: Vec<u32> = Vec::new();
                for (len, _, _) in &paths {
                    if distinct.last() != Some(len) {
                        distinct.push(*len);
                    }
                }
                match distinct
                    .get((k as usize).min(distinct.len()).saturating_sub(1))
                    .copied()
                {
                    Some(cut) => paths
                        .into_iter()
                        .filter(|(l, _, _)| *l <= cut)
                        .map(|(_, n, e)| (n, e))
                        .collect(),
                    None => Vec::new(),
                }
            } else {
                paths
                    .into_iter()
                    .take(k as usize)
                    .map(|(_, n, e)| (n, e))
                    .collect()
            };
            for (nodes, edges) in selected {
                keep.push(row);
                ends.push(end);
                if track {
                    push_path(
                        batch,
                        row,
                        &nodes,
                        &edges,
                        &mut bufs.values,
                        &mut bufs.offsets,
                        &mut bufs.edges,
                        &mut bufs.edge_offsets,
                    );
                }
            }
        }
    }
    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(ends));
    let mut out = Batch::of(slots);
    if track {
        out.lineage = Some(Lineage {
            values: bufs.values,
            offsets: bufs.offsets,
            edges: bufs.edges,
            edge_offsets: bufs.edge_offsets,
        });
    }
    out
}

/// Enumerate every TRAIL (no edge reused) from the source, recording each at every
/// length in `min..=max` into `per_end` keyed by its current endpoint, with the full
/// node/edge chain. `used` holds the on-path edge ids; `node_stack`/`edge_stack` the
/// chain source..`v`. See [`shortest_k_path`].
#[allow(clippy::too_many_arguments)]
fn collect_trails(
    store: &Store,
    v: u32,
    len: u32,
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    edge_pred: Option<&Expr>,
    used: &mut Vec<u32>,
    node_stack: &mut Vec<u32>,
    edge_stack: &mut Vec<u32>,
    per_end: &mut TrailsByEnd,
) {
    if len >= min {
        per_end
            .entry(v)
            .or_default()
            .push((len, node_stack.clone(), edge_stack.clone()));
    }
    if len == max {
        return;
    }
    // OUT then IN adjacency, matching the walkers' emission order; a trail forbids
    // reusing an edge, which also bounds the recursion (depth <= edge count).
    let out: &[crate::store::Adj] = if matches!(dir, Dir::Out | Dir::Both) {
        store.out(v)
    } else {
        &[]
    };
    let inc: &[crate::store::Adj] = if matches!(dir, Dir::In | Dir::Both) {
        store.inc(v)
    } else {
        &[]
    };
    let drop_loop = matches!(dir, Dir::Both);
    for (is_inc, a) in out
        .iter()
        .map(|a| (false, a))
        .chain(inc.iter().map(|a| (true, a)))
    {
        if !edge_carries_wanted(store, a, want)
            || used.contains(&a.eid)
            || !edge_pred_ok(edge_pred, store, a.eid)
        {
            continue;
        }
        if is_inc && drop_loop && a.nbr == v {
            continue;
        }
        used.push(a.eid);
        node_stack.push(a.nbr);
        edge_stack.push(a.eid);
        collect_trails(
            store,
            a.nbr,
            len + 1,
            min,
            max,
            dir,
            want,
            edge_pred,
            used,
            node_stack,
            edge_stack,
            per_end,
        );
        used.pop();
        node_stack.pop();
        edge_stack.pop();
    }
}

/// The single shortest path start..node via the FIRST predecessor of each node (the
/// BFS-tree parent) — the representative `ANY SHORTEST` keeps. Returns the node chain
/// (start..node inclusive) and its edge chain. `node == start` gives `([start], [])`.
fn first_pred_chain(
    node: u32,
    start: u32,
    preds: &FnvMap<u32, Vec<(u32, u32)>>,
) -> (Vec<u32>, Vec<u32>) {
    let mut chain = vec![node];
    let mut echain = Vec::new();
    let mut cur = node;
    while cur != start {
        let (prev, e) = preds[&cur][0];
        echain.push(e);
        cur = prev;
        chain.push(cur);
    }
    chain.reverse();
    echain.reverse();
    (chain, echain)
}

/// Every distinct shortest path start..node through the predecessor DAG, each as a
/// (node chain start..node, edge chain) pair. Exponential on a wide lattice — the
/// same cost core's `enumerate_shortest_paths` pays; no case in scope hits it.
fn enumerate_shortest_paths(
    node: u32,
    start: u32,
    preds: &FnvMap<u32, Vec<(u32, u32)>>,
) -> Vec<(Vec<u32>, Vec<u32>)> {
    if node == start {
        return vec![(vec![start], Vec::new())];
    }
    let mut out = Vec::new();
    if let Some(ps) = preds.get(&node) {
        for &(prev, e) in ps {
            for (mut chain, mut echain) in enumerate_shortest_paths(prev, start, preds) {
                chain.push(node);
                echain.push(e);
                out.push((chain, echain));
            }
        }
    }
    out
}

/// Append one shortest-path row's lineage: the input row's carried path (ending at
/// `start`) followed by the reconstructed `start..node` chain and its edges.
#[allow(clippy::too_many_arguments)]
/// The four parallel buffers a [`Lineage`] is assembled from, seeded with the
/// leading `0` offset each side needs. Shared by the var-length DFS to accumulate a
/// per-emitted-row path.
struct PathBufs {
    values: Vec<Value>,
    offsets: Vec<usize>,
    edges: Vec<Value>,
    edge_offsets: Vec<usize>,
}

impl PathBufs {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            offsets: vec![0],
            edges: Vec::new(),
            edge_offsets: vec![0],
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_path(
    batch: &Batch,
    row: usize,
    chain: &[u32],
    echain: &[u32],
    path_values: &mut Vec<Value>,
    path_offsets: &mut Vec<usize>,
    path_edges: &mut Vec<Value>,
    path_edge_offsets: &mut Vec<usize>,
) {
    let lin = batch.lineage.as_ref().expect("track");
    path_values.extend_from_slice(lin.path_at(row));
    for &n in &chain[1..] {
        path_values.push(Value::Num(f64::from(n)));
    }
    path_offsets.push(path_values.len());
    path_edges.extend_from_slice(lin.edges_at(row));
    for &e in echain {
        path_edges.push(Value::Num(f64::from(e)));
    }
    path_edge_offsets.push(path_edges.len());
}

/// Elementwise `l OP r` over two already-evaluated columns — the general arithmetic
/// body, shared by `Expr::Arith` and its scalar fast path's non-numeric fallback.
/// Raw f64 when both are `Col::Num`; otherwise per-cell via the value contract (a
/// NULL / non-numeric operand → NULL, a temporal operand → `temporal_arith`). Div/Rem
/// by a zero divisor (the RIGHT operand) throws, matching core's DataException.
fn arith_general(op: crate::ir::ArithOp, l: &Col, r: &Col) -> Result<Col, String> {
    use crate::ir::ArithOp::{Add, Div, Mul, Rem, Sub};
    if let (Col::Num(xs), Col::Num(ys)) = (l, r) {
        let n = xs.len().min(ys.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (x, y) = (xs[i], ys[i]);
            if matches!(op, Div | Rem) && y == 0.0 {
                return Err("division by zero".into());
            }
            out.push(match op {
                Add => x + y,
                Sub => x - y,
                Mul => x * y,
                Div => x / y,
                Rem => x % y,
            });
        }
        return Ok(Col::Num(out));
    }
    let n = l.len().min(r.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = l.value_at(i);
        let b = r.value_at(i);
        let v = if matches!(a, Value::Temporal(_)) || matches!(b, Value::Temporal(_)) {
            if a.is_null() || b.is_null() {
                Value::Null
            } else {
                temporal_arith(op, &a, &b)?
            }
        } else {
            match (value::num_of(&a), value::num_of(&b)) {
                (Some(x), Some(y)) => {
                    if matches!(op, Div | Rem) && y == 0.0 {
                        return Err("division by zero".into());
                    }
                    Value::Num(match op {
                        Add => x + y,
                        Sub => x - y,
                        Mul => x * y,
                        Div => x / y,
                        Rem => x % y,
                    })
                }
                // A NULL operand → NULL (three-valued). A NON-null NON-numeric operand
                // (string/bool/list/record) is a DATA EXCEPTION — arithmetic never
                // implicitly coerces; use an explicit CAST (`CAST('1' AS INT) * n`). This
                // is core's SQL-style rule (`'abc' + 1` throws; `1 + null` is null).
                _ if a.is_null() || b.is_null() => Value::Null,
                _ => return Err("arithmetic requires a number".into()),
            }
        };
        out.push(v);
    }
    Ok(Col::Gen(out))
}

/// Evaluate `expr` over every row of `batch`, producing a column.
fn eval(expr: &Expr, store: &Store, batch: &Batch) -> Result<Col, String> {
    // No rows → no values to produce, and nothing to evaluate: a constant faulting
    // expression (`1/0` under `… LIMIT 0 RETURN 1/0`) must not error over an empty
    // batch. Short-circuit before any per-expression work.
    if batch.rows() == 0 {
        return Ok(Col::Gen(Vec::new()));
    }
    Ok(match expr {
        Expr::Slot(n) => batch.slot(*n).clone(),
        Expr::Lit(v) => broadcast(v.clone(), batch.rows()),
        // A `Param` must be substituted by `bind::bind_params` before eval; if one
        // survives, fail loudly rather than mis-evaluate (the safety net).
        Expr::Param(name) => {
            return Err(format!(
                "unbound parameter `${name}` (internal: not bound before evaluation)"
            ))
        }
        Expr::Prop { slot, key } => read_property(store, batch.slot(*slot), key),
        // `<base>.key` — evaluate the base to a column, then read the field/property
        // from it (the general form of `Prop`).
        Expr::Field { base, key } => {
            let col = eval(base, store, batch)?;
            read_property(store, &col, key)
        }
        // `base[index]` — 0-based list element or record/map field. Out of range /
        // negative / non-integer index → NULL; null-safe. Mirrors core.
        //
        // Special case: `nodes(p)[i]` / relationships(p)[i]` (an Index over a path
        // accessor) must keep the ELEMENT typing so a following `.prop` resolves the
        // node/edge property (`edges(p)[0].w`). The path lists carry ids as `Num`,
        // which a generic list-index would flatten to an untyped scalar. Emit a typed
        // `Col::Nodes`/`Col::Edges` instead (out-of-range → `u32::MAX` null sentinel).
        Expr::Index { base, index, .. }
            if matches!(
                base.as_ref(),
                Expr::PathAccess {
                    part: crate::ir::PathPart::Nodes | crate::ir::PathPart::Relationships
                }
            ) =>
        {
            let is_nodes = matches!(
                base.as_ref(),
                Expr::PathAccess {
                    part: crate::ir::PathPart::Nodes
                }
            );
            let icol = eval(index, store, batch)?;
            let ids: Vec<u32> = match &batch.lineage {
                Some(lin) => (0..batch.rows())
                    .map(|i| {
                        let elems = if is_nodes {
                            lin.path_at(i)
                        } else {
                            lin.edges_at(i)
                        };
                        match icol.value_at(i) {
                            Value::Num(n)
                                if n >= 0.0 && n.fract() == 0.0 && (n as usize) < elems.len() =>
                            {
                                match elems[n as usize] {
                                    Value::Num(x) => x as u32,
                                    _ => u32::MAX,
                                }
                            }
                            _ => u32::MAX,
                        }
                    })
                    .collect(),
                None => vec![u32::MAX; batch.rows()],
            };
            if is_nodes {
                Col::Nodes(ids)
            } else {
                Col::Edges(ids)
            }
        }
        Expr::Index { base, index, elem } => {
            let bcol = eval(base, store, batch)?;
            let icol = eval(index, store, batch)?;
            // Index into the per-row list/record/map → the element value (or NULL).
            let at = |i: usize| match bcol.value_at(i) {
                Value::List(items) => match icol.value_at(i) {
                    Value::Num(n) if n >= 0.0 && n.fract() == 0.0 && (n as usize) < items.len() => {
                        items[n as usize].clone()
                    }
                    _ => Value::Null,
                },
                Value::Record(fields) => match icol.value_at(i) {
                    Value::Str(k) => fields
                        .iter()
                        .find(|(fk, _)| *fk == k)
                        .map_or(Value::Null, |(_, v)| v.clone()),
                    _ => Value::Null,
                },
                Value::Map(entries) => match icol.value_at(i) {
                    Value::Str(k) => entries
                        .iter()
                        .find(|(ek, _)| matches!(ek, Value::Str(s) if *s == k))
                        .map_or(Value::Null, |(_, v)| v.clone()),
                    _ => Value::Null,
                },
                _ => Value::Null,
            };
            match elem {
                // A group-variable list element keeps NODE/EDGE typing so a following
                // `.prop` resolves — mirror the path-subscript case: emit a typed
                // `Col::Nodes`/`Col::Edges` (out-of-range / non-node → u32::MAX null).
                crate::ir::ElemKind::Node | crate::ir::ElemKind::Edge => {
                    let ids: Vec<u32> = (0..batch.rows())
                        .map(|i| match at(i) {
                            Value::Num(x) if x >= 0.0 && x.fract() == 0.0 => x as u32,
                            _ => u32::MAX,
                        })
                        .collect();
                    if matches!(elem, crate::ir::ElemKind::Node) {
                        Col::Nodes(ids)
                    } else {
                        Col::Edges(ids)
                    }
                }
                crate::ir::ElemKind::Plain => Col::Gen((0..batch.rows()).map(at).collect()),
            }
        }
        Expr::Path => match &batch.lineage {
            // Each row's path as a List of node ids; NULL when the plan tracks no
            // lineage (which `needs_lineage` prevents when Path is actually read).
            Some(lin) => Col::Gen(
                (0..batch.rows())
                    .map(|i| Value::List(lin.path_at(i).to_vec()))
                    .collect(),
            ),
            None => Col::Gen(vec![Value::Null; batch.rows()]),
        },
        Expr::GremlinPath { ends_on_edge, bys } => match &batch.lineage {
            Some(lin) => Col::Gen(
                (0..batch.rows())
                    .map(|i| {
                        let nodes = lin.path_at(i);
                        let edges = lin.edges_at(i);
                        // Interleave v0,e0,v1,e1,… ; each entry is (id, is_edge).
                        let mut elems: Vec<(u32, bool)> = Vec::new();
                        for j in 0..edges.len() {
                            if let (Some(Value::Num(nv)), Some(Value::Num(ev))) =
                                (nodes.get(j), edges.get(j))
                            {
                                elems.push((*nv as u32, false));
                                elems.push((*ev as u32, true));
                            }
                        }
                        // The final vertex, unless the path stops on the edge (`outE`
                        // with no following `inV` — the recorded target is premature).
                        if !ends_on_edge {
                            if let Some(Value::Num(nv)) = nodes.get(edges.len()) {
                                elems.push((*nv as u32, false));
                            }
                        }
                        let out: Vec<Value> = elems
                            .iter()
                            .enumerate()
                            .map(|(p, &(id, is_edge))| {
                                let by = if bys.is_empty() {
                                    &crate::ir::GPathBy::Element
                                } else {
                                    &bys[p % bys.len()]
                                };
                                render_gpath_elem(store, id, is_edge, by)
                            })
                            .collect();
                        Value::List(out)
                    })
                    .collect(),
            ),
            None => Col::Gen(vec![Value::Null; batch.rows()]),
        },
        Expr::PathAccess { part } => {
            use crate::ir::PathPart;
            match &batch.lineage {
                Some(lin) => Col::Gen(
                    (0..batch.rows())
                        .map(|i| {
                            let nodes = lin.path_at(i);
                            let edges = lin.edges_at(i);
                            match part {
                                PathPart::Nodes => Value::List(nodes.to_vec()),
                                PathPart::Relationships => Value::List(edges.to_vec()),
                                // Hops == number of relationships.
                                PathPart::Length => Value::Num(edges.len() as f64),
                                PathPart::Elements => {
                                    // n0, e0, n1, e1, …, nk
                                    let mut items = Vec::with_capacity(nodes.len() + edges.len());
                                    for (j, node) in nodes.iter().enumerate() {
                                        items.push(node.clone());
                                        if let Some(e) = edges.get(j) {
                                            items.push(e.clone());
                                        }
                                    }
                                    Value::List(items)
                                }
                            }
                        })
                        .collect(),
                ),
                None => Col::Gen(vec![Value::Null; batch.rows()]),
            }
        }
        Expr::Not(inner) => {
            let c = eval(inner, store, batch)?;
            map_bool(&c, |b| b.map(|x| !x))?
        }
        Expr::And(l, r) => zip_bool(store, batch, l, r, |a, b| match (a, b) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        })?,
        Expr::Or(l, r) => zip_bool(store, batch, l, r, |a, b| match (a, b) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        })?,
        // Three-valued XOR: both known → `a != b`; any UNKNOWN operand → UNKNOWN.
        Expr::Xor(l, r) => zip_bool(store, batch, l, r, |a, b| match (a, b) {
            (Some(x), Some(y)) => Some(x != y),
            _ => None,
        })?,
        Expr::Compare { op, left, right } => {
            let l = eval(left, store, batch)?;
            let r = eval(right, store, batch)?;
            compare(*op, &l, &r)
        }
        Expr::In { needle, haystack } => {
            // Runtime three-valued membership (a literal list desugars to an
            // OR-chain instead; this matches its semantics). Per row: TRUE if any
            // element equals the needle; else UNKNOWN (NULL) if the needle or any
            // element is null (the answer can't be decided); else FALSE. A
            // non-list haystack is NULL.
            let nd = eval(needle, store, batch)?;
            let hs = eval(haystack, store, batch)?;
            let n = batch.rows();
            let out: Vec<Value> = (0..n)
                .map(|i| {
                    let needle = nd.value_at(i);
                    let Value::List(items) = hs.value_at(i) else {
                        return Value::Null;
                    };
                    let mut saw_unknown = needle.is_null();
                    for el in items.iter() {
                        if el.is_null() || needle.is_null() {
                            saw_unknown = true;
                        } else if value::equals(&needle, el) {
                            return Value::Bool(true);
                        }
                    }
                    if saw_unknown {
                        Value::Null
                    } else {
                        Value::Bool(false)
                    }
                })
                .collect();
            Col::Gen(out)
        }
        Expr::Arith { op, left, right } => {
            // f64 math via the value contract's `as_num` (finite Num only); any
            // NULL / non-numeric / non-finite operand OR result yields NULL. When
            // either operand is a temporal, `temporal_arith` takes over (and may
            // THROW on a result out of the representable range).
            use crate::ir::ArithOp::{Add, Div, Mul, Rem, Sub};
            // Scalar-literal fast path: `col OP num` / `num OP col`. Evaluate ONLY the
            // non-literal operand and fold the constant into the loop — never
            // materializing an n-length broadcast column for the literal. A chain like
            // `age * 2 + 1` then costs one gather + two scalar passes instead of two
            // 8 MB constant columns plus a boxed intermediate; at 1M that alloc traffic
            // was the whole gap (proj/arith 0.55x). Semantics match the general arm
            // below: div/rem by a zero DIVISOR throws (the divisor is the RIGHT
            // operand), every other f64 result is kept.
            let lit_num = |e: &Expr| match e {
                Expr::Lit(Value::Num(t)) if t.is_finite() => Some(*t),
                _ => None,
            };
            let scalar = match (lit_num(left), lit_num(right)) {
                (_, Some(t)) => Some((t, false)), // col OP num (num is the divisor)
                (Some(t), None) => Some((t, true)), // num OP col (col is the divisor)
                _ => None,
            };
            if let Some((t, num_on_left)) = scalar {
                let other = if num_on_left { right } else { left };
                let col = eval(other, store, batch)?;
                if let Col::Num(xs) = col {
                    let mut out = Vec::with_capacity(xs.len());
                    if matches!(op, Div | Rem) && num_on_left {
                        // num OP col → the COLUMN is the divisor; a zero cell throws.
                        for &x in &xs {
                            if x == 0.0 {
                                return Err("division by zero".into());
                            }
                            out.push(if matches!(op, Div) { t / x } else { t % x });
                        }
                    } else if matches!(op, Div | Rem) {
                        // col OP num → the LITERAL is the divisor; throw once if zero.
                        if t == 0.0 {
                            return Err("division by zero".into());
                        }
                        for &x in &xs {
                            out.push(if matches!(op, Div) { x / t } else { x % t });
                        }
                    } else {
                        for &x in &xs {
                            let (a, b) = if num_on_left { (t, x) } else { (x, t) };
                            out.push(match op {
                                Add => a + b,
                                Sub => a - b,
                                Mul => a * b,
                                _ => unreachable!(),
                            });
                        }
                    }
                    return Ok(Col::Num(out));
                }
                // The non-literal side is not a raw Num column (a null / boxed / temporal
                // operand): reuse the evaluated `col` and a broadcast literal through the
                // general loop rather than re-evaluating.
                let lit_col = broadcast(Value::Num(t), col.len());
                let (l, r) = if num_on_left {
                    (lit_col, col)
                } else {
                    (col, lit_col)
                };
                return arith_general(*op, &l, &r);
            }
            let l = eval(left, store, batch)?;
            let r = eval(right, store, batch)?;
            return arith_general(*op, &l, &r);
        }
        Expr::Call { name, args } => {
            // A call that is a pure function of one dict column (`upper(city)`, …) is
            // computed per distinct value, not per row. (Element functions take a Slot
            // arg, not a Prop, so `sole_prop_ref` rejects them — no conflict below.)
            if let Some(col) = try_eval_dict_scalar(expr, store, batch) {
                return Ok(col);
            }
            // `element_id(node|edge)` → the element's PRESERVED external id string.
            if name == "element_id" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::Num(id) if matches!(arg, Col::Nodes(_)) => {
                            store.node_ext_id(id as u32).map_or(Value::Null, Value::Str)
                        }
                        Value::Num(eid) if matches!(arg, Col::Edges(_)) => store
                            .edge_ext_id(eid as u32)
                            .map_or(Value::Null, Value::Str),
                        _ => Value::Null,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `type(edge)` needs the store + the edge identity (an eid), so it is
            // handled here (off the evaluated arg column), not in `call_scalar`.
            if name == "type" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::Num(eid) if matches!(arg, Col::Edges(_)) => store
                            .edge_type_name(eid as u32)
                            .map_or(Value::Null, |t| Value::Str(t.into())),
                        _ => Value::Null,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `element_label(node|edge)` → a SINGLE label string (Gremlin `label()`):
            // a vertex's label, an edge's type. Not user-callable from GQL (which has
            // list-valued `labels()` and `type()`); emitted only by the Gremlin
            // front-end. A vertex with several labels yields the first in the store's
            // canonical (sorted) order, consistent with GQL `labels()`; a vertex with
            // no label yields Null.
            if name == "element_label" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                // Large node frontier (`V().label()`): resolve labels through the
                // store's one-pass forward map (O(total membership) + O(1)/row) rather
                // than probing every label bucket per node (O(labels·log n)/row, cache-
                // hostile). Small frontiers keep the per-node path — inverting the whole
                // store would cost more than a handful of probes.
                if let Col::Nodes(ids) = &arg {
                    if n >= store.node_count() / 4 {
                        let map = store.min_label_map();
                        let (names, code_of) = &*map;
                        // Gather codes in one pass; if EVERY node is labelled, emit a
                        // typed `Col::Str` — no per-row `Value::Str` box, and the JSON
                        // writer takes its string fast path. A single unlabelled node
                        // (needs a NULL, which `Col::Str` cannot hold) falls to `Col::Gen`.
                        let mut labels: Vec<std::sync::Arc<str>> = Vec::with_capacity(ids.len());
                        let mut all_labelled = true;
                        for &id in ids {
                            match code_of.get(id as usize) {
                                Some(&c) if c != u32::MAX => labels.push(names[c as usize].clone()),
                                _ => {
                                    all_labelled = false;
                                    break;
                                }
                            }
                        }
                        if all_labelled {
                            return Ok(Col::Str(labels));
                        }
                        let out: Vec<Value> = ids
                            .iter()
                            .map(|&id| match code_of.get(id as usize) {
                                Some(&c) if c != u32::MAX => Value::Str(names[c as usize].clone()),
                                _ => Value::Null,
                            })
                            .collect();
                        return Ok(Col::Gen(out));
                    }
                }
                // Intern each distinct label ONCE for the whole column, so a big
                // `V().label()` frontier allocates one Arc per label, not per row.
                let mut cache: Vec<(&str, std::sync::Arc<str>)> = Vec::new();
                let mut out: Vec<Value> = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(match arg.value_at(i) {
                        Value::Num(id) if matches!(arg, Col::Nodes(_)) => {
                            match store.min_label_name(id as u32) {
                                Some(nm) => {
                                    let arc = match cache.iter().find(|(c, _)| *c == nm) {
                                        Some((_, a)) => a.clone(),
                                        None => {
                                            let a: std::sync::Arc<str> = std::sync::Arc::from(nm);
                                            cache.push((nm, a.clone()));
                                            a
                                        }
                                    };
                                    Value::Str(arc)
                                }
                                None => Value::Null,
                            }
                        }
                        Value::Num(eid) if matches!(arg, Col::Edges(_)) => store
                            .edge_type_name(eid as u32)
                            .map_or(Value::Null, |t| Value::Str(t.into())),
                        _ => Value::Null,
                    });
                }
                return Ok(Col::Gen(out));
            }
            // `element_map(element[, 'k1', …])` → Gremlin `elementMap()`: core's FLAT
            // shape — `{id, label, <props…>}` for a node, plus `IN`/`OUT` endpoint
            // stubs for an edge — where `label` is SINGULAR (the first label / edge
            // type) and the present properties are flattened alongside the tokens
            // (so a property named `id`/`label` would shadow one; that's the lossy
            // flat form, distinct from the nested `{id, labels, properties}` render).
            // An optional trailing key list filters the properties. Gremlin-only.
            if name == "element_map" {
                let filter: Vec<String> = args[1..]
                    .iter()
                    .filter_map(|e| match e {
                        Expr::Lit(Value::Str(s)) => Some(s.to_string()),
                        _ => None,
                    })
                    .collect();
                // The first (sorted) label of a node, or an edge's type.
                let node_label = |id: u32| -> Value {
                    let mut ls = store.labels_of(id);
                    ls.sort();
                    ls.into_iter()
                        .next()
                        .map_or(Value::Null, |l| Value::Str(l.into()))
                };
                let node_id = |id: u32| store.node_ext_id(id).map_or(Value::Null, Value::Str);
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                // Resolve the node columns ONCE (sorted, present-filtered per node below)
                // instead of re-cloning+sorting the key list and HashMap-probing per node.
                let node_cols = resolve_node_cols(store, &filter);
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        let mut entries: Vec<(Value, Value)> = Vec::new();
                        match arg.value_at(i) {
                            Value::Num(id) if matches!(arg, Col::Nodes(_)) => {
                                let id = id as u32;
                                let ni = id as usize;
                                entries.push((Value::Str("id".into()), node_id(id)));
                                entries.push((Value::Str("label".into()), node_label(id)));
                                for (k, col) in &node_cols {
                                    if col.present_at(ni) {
                                        entries.push((Value::Str(Arc::clone(k)), col.read(ni)));
                                    }
                                }
                            }
                            Value::Num(eid) if matches!(arg, Col::Edges(_)) => {
                                let eid = eid as u32;
                                entries.push((
                                    Value::Str("id".into()),
                                    store.edge_ext_id(eid).map_or(Value::Null, Value::Str),
                                ));
                                entries.push((
                                    Value::Str("label".into()),
                                    store
                                        .edge_type_name(eid)
                                        .map_or(Value::Null, |t| Value::Str(t.into())),
                                ));
                                if let Some((src, dst)) = store.edge_endpoints(eid) {
                                    let stub = |v: u32| {
                                        Value::Map(Arc::new(vec![
                                            (Value::Str("id".into()), node_id(v)),
                                            (Value::Str("label".into()), node_label(v)),
                                        ]))
                                    };
                                    // Core: IN is the destination, OUT the source.
                                    entries.push((Value::Str("IN".into()), stub(dst)));
                                    entries.push((Value::Str("OUT".into()), stub(src)));
                                }
                                let keys = if filter.is_empty() {
                                    store.edge_prop_keys()
                                } else {
                                    filter.clone()
                                };
                                let mut props: Vec<(String, Value)> = keys
                                    .into_iter()
                                    .filter(|k| store.has_edge_prop(eid, k))
                                    .map(|k| {
                                        let v = store.edge_prop(eid, &k);
                                        (k, v)
                                    })
                                    .collect();
                                props.sort_by(|a, b| a.0.cmp(&b.0));
                                for (k, v) in props {
                                    entries.push((Value::Str(k.into()), v));
                                }
                            }
                            _ => return Value::Null,
                        }
                        Value::Map(Arc::new(entries))
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `value_map(element[, 'k1', …])` → Gremlin `valueMap()`: a Value::Map of
            // the element's PRESENT properties (no id/label tokens), with SCALAR
            // values (core's `propertyMap()`, not built here, is the list-wrapped
            // form). An optional trailing key list filters; no keys = every present
            // property. Keys are sorted (the engine's element-map convention; map key
            // order is set-based per policy). Gremlin-only — not in the GQL whitelist.
            if name == "value_map" || name == "property_map" {
                // `property_map` is `value_map` with each value wrapped in a single-
                // element LIST (a TinkerPop property is multi-valued; lenke is single).
                let wrap = name == "property_map";
                // The filter keys are constant string literals after the element arg.
                let filter: Vec<String> = args[1..]
                    .iter()
                    .filter_map(|e| match e {
                        Expr::Lit(Value::Str(s)) => Some(s.to_string()),
                        _ => None,
                    })
                    .collect();
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                // Node columns resolved ONCE (sorted); the node arm reads straight from
                // them, skipping the per-node key clone+sort and per-key HashMap probes.
                let node_cols = resolve_node_cols(store, &filter);
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        // The node arm emits its (already-sorted) pairs directly; only the
                        // edge arm builds an unsorted `pairs` that needs the sort below.
                        let mut pairs: Vec<(String, Value)> = match arg.value_at(i) {
                            Value::Num(id) if matches!(arg, Col::Nodes(_)) => {
                                let ni = id as usize;
                                return {
                                    let entries: Vec<(Value, Value)> = node_cols
                                        .iter()
                                        .filter(|(_, col)| col.present_at(ni))
                                        .map(|(k, col)| {
                                            let v = col.read(ni);
                                            let v = if wrap { Value::List(vec![v]) } else { v };
                                            (Value::Str(Arc::clone(k)), v)
                                        })
                                        .collect();
                                    Value::Map(Arc::new(entries))
                                };
                            }
                            Value::Num(eid) if matches!(arg, Col::Edges(_)) => {
                                let eid = eid as u32;
                                let keys = if filter.is_empty() {
                                    store.edge_prop_keys()
                                } else {
                                    filter.clone()
                                };
                                keys.into_iter()
                                    .filter(|k| store.has_edge_prop(eid, k))
                                    .map(|k| {
                                        let v = store.edge_prop(eid, &k);
                                        (k, v)
                                    })
                                    .collect()
                            }
                            _ => return Value::Null,
                        };
                        pairs.sort_by(|a, b| a.0.cmp(&b.0));
                        Value::Map(Arc::new(
                            pairs
                                .into_iter()
                                .map(|(k, v)| {
                                    let v = if wrap { Value::List(vec![v]) } else { v };
                                    (Value::Str(k.into()), v)
                                })
                                .collect(),
                        ))
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `path_nodes(path)` → Gremlin `path()` over a vertex-hop chain: render
            // each node id in the lineage path as its element map, so the path is a
            // list of vertex elements (not bare ids). The argument is `Expr::Path`
            // (a per-row list of node-id Nums); a Null row (no lineage) stays Null.
            // Gremlin-only — not in the GQL whitelist.
            if name == "path_nodes" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::List(ids) => Value::List(
                            ids.into_iter()
                                .map(|v| match v {
                                    Value::Num(id) => node_result_value(store, id as u32),
                                    other => other,
                                })
                                .collect(),
                        ),
                        other => other,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `path_values(path, 'k')` → Gremlin `path().by('k')`: render each path
            // element as its `k` property instead of the whole vertex element map.
            if name == "path_values" {
                let arg = eval(&args[0], store, batch)?;
                let key = match &args[1] {
                    Expr::Lit(Value::Str(s)) => s.clone(),
                    _ => return Err("path().by(...) key must be a literal string".into()),
                };
                let n = batch.rows();
                // Sentinel keys from `path().by(id|label)`: the element's ext-id / label.
                let map_elem = |id: u32| -> Value {
                    match key.as_ref() {
                        "\u{0}id" => store.node_ext_id(id).map_or(Value::Null, Value::Str),
                        "\u{0}label" => store
                            .labels_of(id)
                            .into_iter()
                            .next()
                            .map_or(Value::Null, |l| Value::Str(l.into())),
                        _ => store.prop(id, &key),
                    }
                };
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::List(ids) => Value::List(
                            ids.into_iter()
                                .map(|v| match v {
                                    Value::Num(id) => map_elem(id as u32),
                                    other => other,
                                })
                                .collect(),
                        ),
                        other => other,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `path_has_dup(path)` → Gremlin `cyclicPath`/`simplePath` support: TRUE if
            // the lineage node path repeats any vertex, FALSE if all distinct. The
            // argument is `Expr::Path` (a per-row list of node-id Nums); a Null row
            // (no lineage) is Null. Gremlin-only — not in the GQL whitelist.
            if name == "path_has_dup" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::List(ids) => {
                            let mut seen: std::collections::HashSet<u64> =
                                std::collections::HashSet::new();
                            let dup = ids.iter().any(|v| match v {
                                Value::Num(id) => !seen.insert(id.to_bits()),
                                _ => false,
                            });
                            Value::Bool(dup)
                        }
                        other => other,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `list_{sum,mean,min,max}(list)` → Gremlin's scope-LOCAL aggregates over
            // a list cell (e.g. after `fold()`): reduce the list's NUMERIC elements
            // (nulls/non-numerics skipped), yielding Null for a list with no number —
            // matching core's `local_num`/`local_extreme` on the numeric case.
            // Gremlin-only. (Mixed numeric+non-numeric lists are the held cross-type
            // territory; here the non-numerics are simply skipped.)
            // `list_count(list)` → Gremlin `count(local)`: the number of local
            // elements (a list's length, or 1 for a scalar cell — core's
            // `local_elems(v).len()`). Gremlin-only.
            if name == "list_count" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::List(items) => Value::Num(items.len() as f64),
                        _ => Value::Num(1.0),
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `list_tail(list, k)` → Gremlin `tail(local, k)`: the LAST k elements of
            // each list cell (a scalar cell is a 1-element list → itself when k>=1,
            // else empty). Gremlin-only.
            if name == "list_tail" {
                let arg = eval(&args[0], store, batch)?;
                let k = match &args[1] {
                    Expr::Lit(Value::Num(n)) => *n as usize,
                    _ => return Err("tail(local, k): k must be a literal integer".into()),
                };
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::List(items) => {
                            let start = items.len().saturating_sub(k);
                            Value::List(items[start..].to_vec())
                        }
                        other => {
                            if k >= 1 {
                                other
                            } else {
                                Value::List(vec![])
                            }
                        }
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `list_range(list, lo, hi)` → Gremlin `range(local, lo, hi)`: the
            // half-open slice `[lo, hi)` of each list cell (a scalar cell is a
            // 1-element list). Gremlin-only.
            if name == "list_range" {
                let arg = eval(&args[0], store, batch)?;
                let bound = |e: &Expr| match e {
                    Expr::Lit(Value::Num(n)) => Ok(*n as usize),
                    _ => Err("range(local, …): bounds must be literal integers".to_string()),
                };
                let lo = bound(&args[1])?;
                let hi = bound(&args[2])?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        let items = match arg.value_at(i) {
                            Value::List(items) => items,
                            other => vec![other],
                        };
                        let a = lo.min(items.len());
                        let b = hi.min(items.len()).max(a);
                        Value::List(items[a..b].to_vec())
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `list_none(value, op, cmp)` → Gremlin `none(pred)`: true iff NO element of
            // `value` (a list cell, or a scalar treated as a 1-element list) satisfies
            // `op(element, cmp)`. Vacuously true over an empty list.
            if name == "list_none" {
                let arg = eval(&args[0], store, batch)?;
                let op = match &args[1] {
                    Expr::Lit(Value::Str(s)) => s.to_string(),
                    _ => return Err("none(pred): internal op tag missing".into()),
                };
                let cmp = match &args[2] {
                    Expr::Lit(v) => v.clone(),
                    _ => return Err("none(pred): bound must be a literal".into()),
                };
                let matches_pred = |el: &Value| -> bool {
                    match op.as_str() {
                        "eq" => value::equals(el, &cmp),
                        "neq" => !value::equals(el, &cmp),
                        "gt" => value::cmp_partial(el, &cmp).is_some_and(std::cmp::Ordering::is_gt),
                        "gte" => {
                            value::cmp_partial(el, &cmp).is_some_and(std::cmp::Ordering::is_ge)
                        }
                        "lt" => value::cmp_partial(el, &cmp).is_some_and(std::cmp::Ordering::is_lt),
                        "lte" => {
                            value::cmp_partial(el, &cmp).is_some_and(std::cmp::Ordering::is_le)
                        }
                        _ => false,
                    }
                };
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        let none_match = match arg.value_at(i) {
                            Value::List(items) => !items.iter().any(&matches_pred),
                            other => !matches_pred(&other),
                        };
                        Value::Bool(none_match)
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `list_skip(list, n)` → Gremlin `skip(local, n)`: each list cell WITHOUT
            // its first n elements. Gremlin-only.
            if name == "list_skip" {
                let arg = eval(&args[0], store, batch)?;
                let k = match &args[1] {
                    Expr::Lit(Value::Num(n)) => *n as usize,
                    _ => return Err("skip(local, n): n must be a literal integer".into()),
                };
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        let items = match arg.value_at(i) {
                            Value::List(items) => items,
                            other => vec![other],
                        };
                        let a = k.min(items.len());
                        Value::List(items[a..].to_vec())
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            if matches!(
                name.as_str(),
                "list_sum" | "list_mean" | "list_min" | "list_max"
            ) {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        let nums: Vec<f64> = match arg.value_at(i) {
                            Value::List(items) => items
                                .iter()
                                .filter_map(|v| match v {
                                    Value::Num(x) => Some(*x),
                                    _ => None,
                                })
                                .collect(),
                            // A scalar cell is a one-element local list.
                            Value::Num(x) => vec![x],
                            _ => Vec::new(),
                        };
                        if nums.is_empty() {
                            return Value::Null;
                        }
                        let v = match name.as_str() {
                            "list_sum" => nums.iter().sum(),
                            "list_mean" => nums.iter().sum::<f64>() / nums.len() as f64,
                            "list_min" => nums.iter().copied().fold(f64::INFINITY, f64::min),
                            _ => nums.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        };
                        Value::Num(v)
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // Element functions need the STORE and the element identity (a node/edge
            // slot), which the pure-value `call_scalar` cannot see — handle them
            // here off the evaluated argument column.
            // `map_keys`/`map_values` → Gremlin `select(Column.keys|values)` on a Map:
            // the entry keys or values AS A LIST, in the Map's current (post-order)
            // order. A non-Map cell passes through.
            if matches!(name.as_str(), "map_keys" | "map_values") {
                let arg = eval(&args[0], store, batch)?;
                let want_keys = name == "map_keys";
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::Map(pairs) => Value::List(
                            pairs
                                .iter()
                                .map(|(k, v)| if want_keys { k.clone() } else { v.clone() })
                                .collect(),
                        ),
                        other => other,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            if matches!(name.as_str(), "keys" | "labels" | "property_names") {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        // A node surfaces as Num(id); its keys / property_names are
                        // the SORTED present property keys, its labels the SORTED
                        // labels — both as string lists (matching core).
                        Value::Num(id) if matches!(arg, Col::Nodes(_)) => {
                            let id = id as u32;
                            let mut items: Vec<Value> = if name == "labels" {
                                let mut ls = store.labels_of(id);
                                ls.sort();
                                ls.into_iter().map(|l| Value::Str(l.into())).collect()
                            } else {
                                store
                                    .prop_keys()
                                    .into_iter()
                                    .filter(|k| store.has_prop(id, k))
                                    .map(|k| Value::Str(k.into()))
                                    .collect()
                            };
                            items.sort_by(value::cmp_total);
                            Value::List(items)
                        }
                        // An edge: `labels(e)` is its label list (type first, then any
                        // secondary labels); `keys`/`property_names` its present edge
                        // property keys, sorted.
                        Value::Num(id) if matches!(arg, Col::Edges(_)) => {
                            let eid = id as u32;
                            let mut items: Vec<Value> = if name == "labels" {
                                store
                                    .edge_labels_of(eid)
                                    .into_iter()
                                    .map(|l| Value::Str(l.into()))
                                    .collect()
                            } else {
                                store
                                    .edge_prop_keys()
                                    .into_iter()
                                    .filter(|k| store.has_edge_prop(eid, k))
                                    .map(|k| Value::Str(k.into()))
                                    .collect()
                            };
                            if name == "labels" {
                                // Edge labels keep TYPE-first order (not sorted).
                            } else {
                                items.sort_by(value::cmp_total);
                            }
                            Value::List(items)
                        }
                        _ => Value::Null,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // Vectorized unary numeric functions that map finite→finite over a raw
            // `Num` column stay a `Num` column (no per-row boxing), so a downstream
            // aggregate/compare keeps the f64 fast path — e.g. `sum(abs(x - k))`.
            if args.len() == 1 {
                if let Some(f) = unary_finite_num_fn(name) {
                    if let Col::Num(xs) = eval(&args[0], store, batch)? {
                        return Ok(Col::Num(xs.iter().map(|&x| f(x)).collect()));
                    }
                    // A non-`Num` arg (nulls / mixed) falls through to the boxed path.
                }
            }
            // Evaluate each argument to a column, then dispatch per row. Arity is
            // validated at parse time, so `call_scalar` can index its args. The row
            // count is the BATCH's, not the min over args — a niladic function
            // (`pi()`, `e()`) has no arg columns yet still yields one value per row.
            let cols = eval_all(args, store, batch)?;
            let n = batch.rows();
            // Reuse ONE argument buffer across rows instead of heap-allocating a fresh
            // `Vec<Value>` per row — a general win for every multi-arg scalar function
            // (concat, substring, replace, …), which otherwise paid `n` allocations.
            let mut buf: Vec<Value> = Vec::with_capacity(cols.len());
            let mut out: Vec<Value> = Vec::with_capacity(n);
            for i in 0..n {
                buf.clear();
                buf.extend(cols.iter().map(|c| c.value_at(i)));
                out.push(call_scalar_checked(name, &buf)?);
            }
            // Stays boxed here — a plain computed projection must not pay the typed-fold
            // cost for no benefit (measured: RETURN trim()/substring() regressed). The SORT
            // path converts a homogeneous key column to typed on demand (see order_page).
            Col::Gen(out)
        }
        Expr::List { items } => {
            // Per row, build a Value::List of each element's value. A VERTEX/EDGE element
            // renders as its element map (render_cell), consistent with a top-level one.
            let cols = eval_all(items, store, batch)?;
            let n = batch.rows();
            Col::Gen(
                (0..n)
                    .map(|i| Value::List(cols.iter().map(|c| render_cell(c, i, store)).collect()))
                    .collect(),
            )
        }
        Expr::Record { fields } => {
            // Per row, evaluate each field then canonicalize into a Value::Record
            // (keys sorted, last-wins) via the value contract.
            let cols = eval_all(fields.iter().map(|(_, e)| e), store, batch)?;
            let n = batch.rows();
            Col::Gen(
                (0..n)
                    .map(|i| {
                        let pairs = fields
                            .iter()
                            .zip(&cols)
                            .map(|((k, _), c)| (Arc::from(k.as_str()), c.value_at(i)))
                            .collect();
                        value::make_record(pairs)
                    })
                    .collect(),
            )
        }
        Expr::MapLit { entries } => {
            // Per row, an insertion-ordered Value::Map with string keys. A VERTEX/EDGE
            // value renders as its element map (via render_cell), not a raw dense id, so
            // a project()/select() map of elements canonicalizes like a top-level one.
            let cols = eval_all(entries.iter().map(|(_, e)| e), store, batch)?;
            let n = batch.rows();
            Col::Gen(
                (0..n)
                    .map(|i| {
                        let pairs = entries
                            .iter()
                            .zip(&cols)
                            .map(|((k, _), c)| {
                                (Value::Str(Arc::from(k.as_str())), render_cell(c, i, store))
                            })
                            .collect();
                        Value::Map(Arc::new(pairs))
                    })
                    .collect(),
            )
        }
        Expr::Case {
            branches,
            otherwise,
        } => {
            // Categorical remap fast path: every branch is `<dict col> = <str literal>`.
            // Precompute code → first-matching-branch once, then map each row by its code
            // instead of evaluating a full compare column per branch. Byte-identical: an
            // absent value / null-sentinel matches no branch (→ ELSE), same as the 3VL
            // compares below.
            if let Some((slot, key, code_to_branch)) = case_dict_lookup(branches, store) {
                if let (Some(Column::Dict { codes, present, .. }), Col::Nodes(ids)) =
                    (store.column(&key), batch.slot(slot))
                {
                    let vals = eval_all(branches.iter().map(|(_, v)| v), store, batch)?;
                    let else_col = otherwise
                        .as_ref()
                        .map(|e| eval(e, store, batch))
                        .transpose()?;
                    let out: Vec<Value> = ids
                        .iter()
                        .enumerate()
                        .map(|(i, &id)| {
                            let bi = (id != u32::MAX && present[id as usize])
                                .then(|| code_to_branch[codes[id as usize] as usize])
                                .flatten();
                            match bi {
                                Some(b) => vals[b].value_at(i),
                                None => else_col.as_ref().map_or(Value::Null, |c| c.value_at(i)),
                            }
                        })
                        .collect();
                    return Ok(Col::Gen(out));
                }
            }
            let conds = eval_all(branches.iter().map(|(c, _)| c), store, batch)?;
            let vals = eval_all(branches.iter().map(|(_, v)| v), store, batch)?;
            let else_col = otherwise
                .as_ref()
                .map(|e| eval(e, store, batch))
                .transpose()?;
            let n = batch.rows();
            let out: Vec<Value> = (0..n)
                .map(|i| {
                    // First branch whose condition is TRUE (three-valued). A non-null
                    // non-boolean condition is a data exception — it is NOT coerced to a
                    // truth value (a WHEN must be a boolean, like WHERE / AND / OR).
                    for (bi, c) in conds.iter().enumerate() {
                        match c.value_at(i) {
                            Value::Bool(true) => return Ok(vals[bi].value_at(i)),
                            Value::Bool(false) | Value::Null => {}
                            _ => return Err(TRUTH_TYPE_ERR.to_string()),
                        }
                    }
                    Ok(else_col.as_ref().map_or(Value::Null, |c| c.value_at(i)))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Col::Gen(out)
        }
        Expr::Cast { target, expr } => {
            // Evaluate the input, then cast per row via the value contract. A
            // failed conversion aborts the whole evaluation (E_INVALID_VALUE) —
            // the read pipeline is fallible precisely so this can throw.
            let col = eval(expr, store, batch)?;
            let t = value::CastTarget::from(*target);
            let mut out = Vec::with_capacity(col.len());
            for i in 0..col.len() {
                out.push(value::cast(&col.value_at(i), t)?);
            }
            Col::Gen(out)
        }
        Expr::IsNull { expr, negated } => {
            // A definite Bool per row (never NULL): `IS NULL` is TRUE exactly when
            // the value is Null; `IS NOT NULL` flips it.
            let col = eval(expr, store, batch)?;
            Col::Bool(
                (0..col.len())
                    .map(|i| col.value_at(i).is_null() != *negated)
                    .collect(),
            )
        }
        Expr::IsLabeled { slot, labels } => {
            // Membership via the label buckets: resolve each label's sorted id bucket
            // ONCE, then binary-search per row (nodes) — no per-row list build or string
            // hashing. Edges compare the type name (rarer path). A non-element is false.
            let node_buckets: Vec<&[u32]> =
                labels.iter().map(|l| store.nodes_with_label(l)).collect();
            match batch.slot(*slot) {
                Col::Nodes(ids) => {
                    // Large frontier (a mid-traversal `hasLabel` after a hop, often WITH
                    // multiplicity): build a membership BITSET once — O(total wanted
                    // membership) — and test each row O(1), instead of N cache-hostile
                    // binary searches into the buckets. Small frontiers keep the probe
                    // (building the bitset would cost more than a few searches).
                    let total_bucket: usize = node_buckets.iter().map(|b| b.len()).sum();
                    if ids.len() >= 1024 && ids.len() >= total_bucket {
                        let mut member = vec![false; store.node_count()];
                        for b in &node_buckets {
                            for &id in *b {
                                member[id as usize] = true;
                            }
                        }
                        Col::Bool(
                            ids.iter()
                                .map(|&id| id != u32::MAX && member[id as usize])
                                .collect(),
                        )
                    } else {
                        Col::Bool(
                            ids.iter()
                                .map(|&id| {
                                    id != u32::MAX
                                        && node_buckets.iter().any(|b| b.binary_search(&id).is_ok())
                                })
                                .collect(),
                        )
                    }
                }
                Col::Edges(eids) => Col::Bool(
                    eids.iter()
                        .map(|&e| {
                            e != u32::MAX
                                && store.edge_type_name(e).is_some_and(|t| labels.contains(&t))
                        })
                        .collect(),
                ),
                other => Col::Bool(vec![false; other.len()]),
            }
        }
        Expr::PropertyExists { slot, key } => {
            // Presence, not value: TRUE iff the element carries a stored value for
            // `key`, FALSE if not — but on a NON-element (the OPTIONAL null sentinel
            // `u32::MAX`, or a computed value) the answer is NULL, matching core's
            // `prop_present` (`_ => Val::Null`). A column with no sentinel keeps the
            // unboxed `Col::Bool` fast path; a sentinel forces the null-carrying `Gen`.
            match batch.slot(*slot) {
                Col::Nodes(ids) if !ids.contains(&u32::MAX) => {
                    Col::Bool(ids.iter().map(|&id| store.has_prop(id, key)).collect())
                }
                Col::Nodes(ids) => Col::Gen(
                    ids.iter()
                        .map(|&id| {
                            if id == u32::MAX {
                                Value::Null
                            } else {
                                Value::Bool(store.has_prop(id, key))
                            }
                        })
                        .collect(),
                ),
                Col::Edges(eids) if !eids.contains(&u32::MAX) => {
                    Col::Bool(eids.iter().map(|&e| store.has_edge_prop(e, key)).collect())
                }
                Col::Edges(eids) => Col::Gen(
                    eids.iter()
                        .map(|&e| {
                            if e == u32::MAX {
                                Value::Null
                            } else {
                                Value::Bool(store.has_edge_prop(e, key))
                            }
                        })
                        .collect(),
                ),
                other => Col::Gen(vec![Value::Null; other.len()]),
            }
        }
        Expr::Exists { body, .. } => {
            // Fast path: a bare vertex-hop existence semi-join (`where(out/in/both)`)
            // only asks "does this row have ANY matching neighbour?" — check the
            // adjacency per row and short-circuit, instead of expanding EVERY neighbour
            // of the whole frontier and back-mapping via a provenance column.
            if let Plan::Expand {
                input,
                from,
                dir,
                edge_label,
                bind_edge: false,
                double_loops: _,
            } = body.as_ref()
            {
                if matches!(**input, Plan::Row) {
                    if let Col::Nodes(ids) = batch.slot(*from) {
                        let want = match want_etypes(store, edge_label) {
                            Ok(w) => w,
                            Err(()) => return Ok(Col::Bool(vec![false; ids.len()])),
                        };
                        return Ok(Col::Bool(
                            ids.iter()
                                .map(|&v| v != u32::MAX && node_has_nbr(store, v, *dir, &want))
                                .collect(),
                        ));
                    }
                }
            }
            // Filtered semijoin: `where(out().has(k,v))` / GQL `EXISTS { (n)->(m:L) WHERE
            // … }` — a CHAIN of `Filter`s over `Expand over Row`. Check the neighbour
            // predicates per source with early-stop, instead of expanding every neighbour
            // of the frontier then filtering + back-mapping. Only SIMPLE leaves on the
            // neighbour (compares / presence / label tests).
            {
                let mut cur = body.as_ref();
                let mut filter_preds: Vec<&Expr> = Vec::new();
                while let Plan::Filter { input, pred } = cur {
                    filter_preds.push(pred);
                    cur = input.as_ref();
                }
                if let Plan::Expand {
                    input,
                    from,
                    dir,
                    edge_label,
                    bind_edge: false,
                    double_loops: _,
                } = cur
                {
                    if !filter_preds.is_empty() && matches!(**input, Plan::Row) {
                        let preds: Option<Vec<NbrPred>> = filter_preds
                            .iter()
                            .map(|p| simple_nbr_preds(p, *from))
                            .collect::<Option<Vec<_>>>()
                            .map(|v| v.into_iter().flatten().collect());
                        if let (Col::Nodes(ids), Some(preds)) = (batch.slot(*from), preds) {
                            let want = match want_etypes(store, edge_label) {
                                Ok(w) => w,
                                Err(()) => return Ok(Col::Bool(vec![false; ids.len()])),
                            };
                            // Raw-column fast path for a lone numeric neighbour compare:
                            // resolve the column ONCE, not per neighbour via `store.prop`.
                            if let [NbrPred::Cmp(k, op, Value::Num(t))] = preds.as_slice() {
                                if let Some(Column::Num { data, present, .. }) = store.column(k) {
                                    return Ok(Col::Bool(
                                        ids.iter()
                                            .map(|&v| {
                                                v != u32::MAX
                                                    && node_has_num_nbr(
                                                        store,
                                                        v,
                                                        *dir,
                                                        &want,
                                                        (data, present),
                                                        (*op, *t),
                                                    )
                                            })
                                            .collect(),
                                    ));
                                }
                            }
                            return Ok(Col::Bool(
                                ids.iter()
                                    .map(|&v| {
                                        v != u32::MAX
                                            && node_has_matching_nbr(store, v, *dir, &want, &preds)
                                    })
                                    .collect(),
                            ));
                        }
                    }
                }
            }
            // Correlated existence: run the sub-pattern over ALL outer rows at once,
            // tagging each with a unique provenance id so surviving sub-rows point
            // back to the outer row they came from. An outer row is TRUE iff at
            // least one sub-row carries its id.
            let n = batch.rows();
            let prov = batch.slots.len(); // provenance rides at the first free slot
            let mut slots = batch.slots.clone();
            slots.push(Col::Num((0..n).map(|i| i as f64).collect()));
            // The body reads no path (EXISTS discards lineage), so seed without one.
            let seed = Batch::of(slots);
            let survivors = pull_body(body, store, &seed)?;
            let mut hit = vec![false; n];
            if let Col::Num(ids) = survivors.slot(prov) {
                for &id in ids {
                    let i = id as usize;
                    if i < n {
                        hit[i] = true;
                    }
                }
            }
            Col::Bool(hit)
        }
        Expr::CountSubquery { body, .. } => {
            // Correlated count: same provenance-tagged sub-run as EXISTS, but TALLY
            // the sub-rows per outer row instead of a boolean any().
            let n = batch.rows();
            let prov = batch.slots.len();
            let mut slots = batch.slots.clone();
            slots.push(Col::Num((0..n).map(|i| i as f64).collect()));
            let seed = Batch::of(slots);
            let survivors = pull_body(body, store, &seed)?;
            let mut counts = vec![0f64; n];
            if let Col::Num(ids) = survivors.slot(prov) {
                for &id in ids {
                    let i = id as usize;
                    if i < n {
                        counts[i] += 1.0;
                    }
                }
            }
            Col::Num(counts)
        }
        Expr::CollectSubquery { body, scalar, .. } => {
            // Correlated collect (Gremlin local(<hop>.fold())): the same provenance-
            // tagged sub-run, gathering `scalar` per outer row into a list (empty when
            // nothing matched). Vertices/edges render as element maps (render_cell).
            let n = batch.rows();
            let prov = batch.slots.len();
            let mut slots = batch.slots.clone();
            slots.push(Col::Num((0..n).map(|i| i as f64).collect()));
            let seed = Batch::of(slots);
            let survivors = pull_body(body, store, &seed)?;
            let vals = eval(scalar, store, &survivors)?;
            let mut out: Vec<Vec<Value>> = vec![Vec::new(); n];
            if let Col::Num(ids) = survivors.slot(prov).clone() {
                for (j, &id) in ids.iter().enumerate() {
                    let i = id as usize;
                    if i < n {
                        out[i].push(render_cell(&vals, j, store));
                    }
                }
            }
            Col::Gen(out.into_iter().map(Value::List).collect())
        }
        Expr::ScalarSubquery { body, scalar, .. } => {
            // Correlated scalar: same provenance-tagged sub-run, but project `scalar`
            // over the surviving sub-rows and return each outer row's single value
            // (NULL when the body matched nothing). A VALUE subquery must return AT
            // MOST one row per outer row — more than one is an error (matching core).
            let n = batch.rows();
            let prov = batch.slots.len();
            let mut slots = batch.slots.clone();
            slots.push(Col::Num((0..n).map(|i| i as f64).collect()));
            let seed = Batch::of(slots);
            let survivors = pull_body(body, store, &seed)?;
            let vals = eval(scalar, store, &survivors)?;
            let mut out = vec![Value::Null; n];
            let mut seen = vec![false; n];
            if let Col::Num(ids) = survivors.slot(prov).clone() {
                for (j, &id) in ids.iter().enumerate() {
                    let i = id as usize;
                    if i < n {
                        if seen[i] {
                            return Err("a VALUE subquery returned more than one row".into());
                        }
                        seen[i] = true;
                        out[i] = vals.value_at(j);
                    }
                }
            }
            Col::Gen(out)
        }
        Expr::UncorrelatedExists { body } => {
            // The body references no outer variable — run it ONCE (a self-contained
            // scan/join/filter plan) and broadcast whether it produced any row.
            let exists = pull(body, store, false)?.rows() > 0;
            Col::Bool(vec![exists; batch.rows()])
        }
        Expr::UncorrelatedCount { body } => {
            // Run the self-contained body once; broadcast its row count.
            let n = pull(body, store, false)?.rows() as f64;
            Col::Num(vec![n; batch.rows()])
        }
        Expr::UncorrelatedScalar { body } => {
            // Run the self-contained body (its own RETURN) once; the VALUE is its
            // single value (NULL if empty, an error if more than one row).
            let b = pull(body, store, false)?;
            let v = match b.rows() {
                0 => Value::Null,
                1 => b.slot(0).value_at(0),
                _ => return Err("a VALUE subquery returned more than one row".into()),
            };
            broadcast(v, batch.rows())
        }
        Expr::GraphPred { op, args, negated } => {
            use crate::ir::GraphPredOp;
            // Each operand as a column; per row, its element IDENTITY (kind + id) or
            // None (a NULL / non-element). The predicate is three-valued: any None
            // operand yields NULL.
            let cols: Vec<Col> = args
                .iter()
                .map(|a| eval(a, store, batch))
                .collect::<Result<_, _>>()?;
            let ident = |c: &Col, i: usize| -> Option<(u8, u32)> {
                match c {
                    Col::Nodes(v) if v[i] != u32::MAX => Some((0, v[i])),
                    Col::Edges(v) if v[i] != u32::MAX => Some((1, v[i])),
                    _ => None,
                }
            };
            let out: Vec<Value> = (0..batch.rows())
                .map(|i| {
                    let idents: Vec<Option<(u8, u32)>> = cols.iter().map(|c| ident(c, i)).collect();
                    let r: Option<bool> = match op {
                        GraphPredOp::IsDirected => match idents[0] {
                            Some((1, _)) => Some(true), // an edge is directed
                            Some(_) => Some(false),     // a node is not
                            None => None,
                        },
                        GraphPredOp::IsSourceOf | GraphPredOp::IsDestinationOf => {
                            match (idents[0], idents[1]) {
                                (Some((0, node)), Some((1, eid))) => {
                                    store.edge_endpoints(eid).map(|(s, d)| {
                                        node == if matches!(op, GraphPredOp::IsSourceOf) {
                                            s
                                        } else {
                                            d
                                        }
                                    })
                                }
                                (None, _) | (_, None) => None,
                                _ => Some(false), // wrong kinds (e.g. edge IS SOURCE OF)
                            }
                        }
                        GraphPredOp::AllDifferent | GraphPredOp::Same => {
                            if idents.iter().any(Option::is_none) {
                                None
                            } else {
                                let all_same = idents.windows(2).all(|w| w[0] == w[1]);
                                let all_diff = (0..idents.len())
                                    .all(|a| (a + 1..idents.len()).all(|b| idents[a] != idents[b]));
                                Some(if matches!(op, GraphPredOp::Same) {
                                    all_same
                                } else {
                                    all_diff
                                })
                            }
                        }
                    };
                    match r.map(|b| b ^ *negated) {
                        Some(b) => Value::Bool(b),
                        None => Value::Null,
                    }
                })
                .collect();
            Col::Gen(out)
        }
    })
}

/// Evaluate an `EXISTS` body against a correlated `seed` batch (the outer rows
/// plus a provenance column). The body is a chain of the operators an EXISTS
/// pattern can contain — `Expand`/`VarLength`/`Filter` — rooted at `Plan::Row`,
/// which yields `seed`. Every operator gathers the whole input row, so the
/// provenance column rides through untouched; the caller reads it off the result.
/// Concatenate several same-shaped batches row-wise (Gremlin `union`'s reconverge).
/// Each output slot is the type-preserving concatenation of that slot across the
/// batches — so a column that is `Col::Nodes` in every branch stays a node frontier
/// (continuable), falling back to `Col::Gen` only when the branch column types
/// differ. Empty input → an empty batch.
fn concat_batches(subs: &[Batch]) -> Batch {
    let Some(first) = subs.first() else {
        return Batch::of(Vec::new());
    };
    let ncols = first.slots.len();
    let cols: Vec<Col> = (0..ncols)
        .map(|j| concat_cols(&subs.iter().map(|b| b.slot(j)).collect::<Vec<_>>()))
        .collect();
    let mut out = Batch::of(cols);
    // Preserve lineage: when every sub carries a sidecar, concatenate them row-wise so
    // path()/GremlinPath survives a union/coalesce branch. (All-or-nothing — a partial
    // set would desync the row alignment.)
    if subs.iter().all(|b| b.lineage.is_some()) {
        let lins: Vec<&crate::batch::Lineage> =
            subs.iter().map(|b| b.lineage.as_ref().unwrap()).collect();
        out.lineage = Some(crate::batch::Lineage::concat(&lins));
    }
    out
}

/// Concatenate columns of (ideally) the same variant. Same variant → keep it and
/// extend the inner vector; mixed variants → materialize every value into `Gen`.
fn concat_cols(cols: &[&Col]) -> Col {
    macro_rules! same {
        ($variant:ident) => {{
            let mut v = Vec::new();
            for c in cols {
                if let Col::$variant(xs) = c {
                    v.extend(xs.iter().cloned());
                } else {
                    return Col::Gen(
                        cols.iter()
                            .flat_map(|c| (0..c.len()).map(|i| c.value_at(i)))
                            .collect(),
                    );
                }
            }
            Col::$variant(v)
        }};
    }
    match cols.first() {
        None => Col::Gen(Vec::new()),
        Some(Col::Nodes(_)) => same!(Nodes),
        Some(Col::Edges(_)) => same!(Edges),
        Some(Col::Num(_)) => same!(Num),
        Some(Col::Bool(_)) => same!(Bool),
        Some(Col::Str(_)) => same!(Str),
        Some(Col::Gen(_)) => Col::Gen(
            cols.iter()
                .flat_map(|c| (0..c.len()).map(|i| c.value_at(i)))
                .collect(),
        ),
    }
}

fn pull_body(plan: &Plan, store: &Store, seed: &Batch) -> Result<Batch, String> {
    Ok(match plan {
        Plan::Row => seed.clone(),
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            bind_edge,
            double_loops,
        } => expand(
            &pull_body(input, store, seed)?,
            store,
            *from,
            *dir,
            edge_label,
            *bind_edge,
            *double_loops,
        ),
        // Edge frontier → endpoint vertex (`inV`/`outV`/`otherV` off a bound edge) —
        // the streamable twin of the main EdgeVertex arm; appends the endpoint slot
        // (Both fans out to two rows). Lineage-free (streaming is `!track`).
        Plan::EdgeVertex {
            input,
            edge_slot,
            which,
        } => {
            let b = pull_body(input, store, seed)?;
            let mut keep: Vec<usize> = Vec::new();
            let mut nodes: Vec<u32> = Vec::new();
            for i in 0..b.rows() {
                let eid = match b.slot(*edge_slot).value_at(i) {
                    Value::Num(x) if x >= 0.0 => x as u32,
                    _ => continue,
                };
                let Some((src, dst)) = store.edge_endpoints(eid) else {
                    continue;
                };
                match which {
                    Dir::Out => {
                        keep.push(i);
                        nodes.push(src);
                    }
                    Dir::In => {
                        keep.push(i);
                        nodes.push(dst);
                    }
                    Dir::Both => {
                        keep.push(i);
                        nodes.push(src);
                        keep.push(i);
                        nodes.push(dst);
                    }
                }
            }
            let mut out = b.gather(&keep);
            out.slots.push(Col::Nodes(nodes));
            out
        }
        Plan::VarLength {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            mode,
            until,
            body_filter,
            double_loops,
        } => var_length(
            &pull_body(input, store, seed)?,
            store,
            *from,
            *dir,
            edge_label,
            *min,
            *max,
            *mode,
            &[],
            None,
            1,
            until.as_deref(),
            body_filter.as_deref(),
            *double_loops,
        )?,
        Plan::Filter { input, pred } => {
            let b = pull_body(input, store, seed)?;
            let mask = eval(pred, store, &b)?;
            let keep: Vec<usize> = match &mask {
                Col::Bool(bs) => (0..bs.len()).filter(|&i| bs[i]).collect(),
                other => (0..other.len())
                    .filter(|&i| other.value_at(i).is_true())
                    .collect(),
            };
            b.gather(&keep)
        }
        // A projection is streamable too (used by the LIMIT short-circuit driver;
        // EXISTS bodies never contain one). Evaluate the items over the sub-frontier.
        Plan::Project { input, items } => {
            let b = pull_body(input, store, seed)?;
            let cols = eval_all(items.iter().map(|(_, e)| e), store, &b)?;
            let mut out = Batch::of(cols);
            out.lineage = b.lineage;
            out
        }
        // Append/overwrite one column — a `choose(...identity())` else arm copies the
        // pass-through element into the reconverge slot inside a Branch body.
        Plan::MapSlot {
            input,
            slot,
            value,
            append,
        } => {
            let mut b = pull_body(input, store, seed)?;
            let col = eval(value, store, &b)?;
            if *append {
                b.slots.push(col);
            } else if *slot < b.slots.len() {
                b.slots[*slot] = col;
            }
            b
        }
        // A reducing body (`fold()`/`count()` inside a union/coalesce branch) folds the
        // sub-frontier; run the general aggregate over the pulled body.
        Plan::Aggregate { input, keys, aggs } => {
            let b = pull_body(input, store, seed)?;
            aggregate(&b, store, keys, aggs)?
        }
        // A per-cell list sort inside a branch body.
        Plan::SortLocal {
            input,
            descending,
            by_key,
        } => {
            let b = pull_body(input, store, seed)?;
            let n = b.rows();
            let sorted: Vec<Value> = (0..n)
                .map(|i| sort_local_cell(b.slot(0).value_at(i), *descending, *by_key))
                .collect();
            let mut slots = b.slots.clone();
            if !slots.is_empty() {
                slots[0] = Col::Gen(sorted);
            }
            Batch::of(slots)
        }
        // An unwind inside a branch body (a union of fold/unfold, etc.).
        Plan::Unwind {
            input,
            list,
            ordinal,
            ..
        } => {
            let b = pull_body(input, store, seed)?;
            let lists = eval(list, store, &b)?;
            let mut keep = Vec::new();
            let mut elems: Vec<Value> = Vec::new();
            let mut ords: Vec<Value> = Vec::new();
            for i in 0..b.rows() {
                let items: Vec<Value> = match lists.value_at(i) {
                    Value::List(v) => v,
                    Value::Null => Vec::new(),
                    scalar => vec![scalar],
                };
                for (j, e) in items.into_iter().enumerate() {
                    keep.push(i);
                    elems.push(e);
                    if let Some((_, one_based)) = ordinal {
                        ords.push(Value::Num((j + usize::from(*one_based)) as f64));
                    }
                }
            }
            let mut slots: Vec<Col> = b.slots.iter().map(|c| c.gather(&keep)).collect();
            slots.push(reunfold_elements(&elems, store));
            if ordinal.is_some() {
                slots.push(Col::Gen(ords));
            }
            Batch::of(slots)
        }
        // Paging / dedup / tail in a Row-seeded tail (a `SET … RETURN … ORDER BY` /
        // `DISTINCT`, or a Gremlin `property(…).values(…).dedup()`). The seed is already
        // materialized, so apply the batch-level operators directly — no streaming cap.
        Plan::OrderPage {
            input,
            keys,
            skip,
            limit,
        } => {
            let b = pull_body(input, store, seed)?;
            order_page(&b, store, keys, *skip, *limit)?
        }
        Plan::Distinct { input } => distinct_batch(pull_body(input, store, seed)?),
        Plan::Tail { input, n } => {
            let b = pull_body(input, store, seed)?;
            let rows = b.rows();
            let start = rows.saturating_sub(*n);
            b.gather(&(start..rows).collect::<Vec<usize>>())
        }
        other => {
            return Err(format!("unsupported operator in EXISTS body: {other:?}"));
        }
    })
}

/// Evaluate several expressions to columns, short-circuiting on the first error.
fn eval_all<'a>(
    exprs: impl IntoIterator<Item = &'a Expr>,
    store: &Store,
    batch: &Batch,
) -> Result<Vec<Col>, String> {
    exprs.into_iter().map(|e| eval(e, store, batch)).collect()
}

/// Dispatch a scalar function over its already-evaluated argument row. Arity is
/// enforced by the parser, so indexing `args` here is safe. NULL / wrong-type
/// arguments yield NULL (no coercion, no throw).
/// The fallible wrapper around [`call_scalar`]: nearly every scalar function is
/// total, but a temporal component accessor of a kind that lacks that component
/// (`_year` of a time, `_hour` of a date) FAULTS with `E_INVALID_VALUE`. A non-null
/// NON-temporal argument (a number or string — never coerced) is ALSO a data
/// exception (matching core); only a NULL arg yields NULL (nullish propagation).
fn call_scalar_checked(name: &str, args: &[Value]) -> Result<Value, String> {
    if matches!(
        name,
        "_year" | "_month" | "_day" | "_hour" | "_minute" | "_second"
    ) {
        return match &args[0] {
            Value::Temporal(t) => date_part(name.trim_start_matches('_'), *t)
                .map(|n| Value::Num(n as f64))
                .ok_or_else(|| {
                    format!(
                        "E_INVALID_VALUE: {} is undefined for this temporal kind",
                        name.trim_start_matches('_')
                    )
                }),
            Value::Null => Ok(Value::Null),
            _ => Err(format!(
                "E_INVALID_VALUE: {}() requires a temporal value (a string is not coerced)",
                name.trim_start_matches('_')
            )),
        };
    }
    // Numeric scalar functions take numbers only — a non-null, non-numeric argument
    // (a string, bool, temporal, list) is a data exception, never coerced (the same
    // SQL rule as arithmetic and the temporal accessors above). `sqrt('1e300')`
    // throws rather than silently returning NULL or coercing the string. A NULL arg
    // still propagates to NULL inside `call_scalar`. Unlike an arithmetic OPERATOR
    // (which propagates null before type-checking), a named function VALIDATES its
    // non-null arguments even beside a null: `atan2(null, duration)` is a type error.
    if matches!(
        name,
        "abs"
            | "sign"
            | "floor"
            | "ceil"
            | "ceiling"
            | "sqrt"
            | "exp"
            | "ln"
            | "log10"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "cot"
            | "degrees"
            | "radians"
            | "round"
            | "log"
            | "power"
            | "mod"
            | "atan2"
    ) && args
        .iter()
        .any(|a| !a.is_null() && !matches!(a, Value::Num(_)))
    {
        return Err(format!(
            "E_INVALID_VALUE: {name}() requires a number (a string is not coerced)"
        ));
    }
    // String / byte scalar functions take strings — a non-null, non-string argument is a
    // data exception, never coerced (the same rule as the numeric functions above; only a
    // NULL arg propagates to NULL). Mixed-arity fns type each position: `left`/`right`/
    // `substring` take a string then number(s); the rest are all-string. `reverse`/`size`/
    // `cardinality` are polymorphic over a string OR a list, so they fault only on neither.
    {
        let (str_pos, num_pos): (&[usize], &[usize]) = match name {
            "upper" | "lower" | "char_length" | "character_length" | "byte_length"
            | "octet_length" => (&[0], &[]),
            "trim" | "btrim" | "ltrim" | "rtrim" => (&[0, 1], &[]),
            "split" | "starts_with" | "ends_with" | "contains" | "regex_match" => (&[0, 1], &[]),
            "replace" => (&[0, 1, 2], &[]),
            "left" | "right" => (&[0], &[1]),
            "substring" => (&[0], &[1, 2]),
            _ => (&[], &[]),
        };
        for &i in str_pos {
            if let Some(a) = args.get(i) {
                if !a.is_null() && !matches!(a, Value::Str(_)) {
                    return Err(format!(
                        "E_INVALID_VALUE: {name}() requires a string (a number is not coerced)"
                    ));
                }
            }
        }
        for &i in num_pos {
            if let Some(a) = args.get(i) {
                if !a.is_null() && !matches!(a, Value::Num(_)) {
                    return Err(format!(
                        "E_INVALID_VALUE: {name}() requires a number (a string is not coerced)"
                    ));
                }
            }
        }
        if matches!(name, "reverse" | "size" | "cardinality")
            && args
                .first()
                .is_some_and(|a| !a.is_null() && !matches!(a, Value::Str(_) | Value::List(_)))
        {
            return Err(format!(
                "E_INVALID_VALUE: {name}() requires a string or list"
            ));
        }
        // `||` concatenation (lowered to `concat`): operands must be homogeneous and
        // concatenable — all strings OR all lists. A NULL operand makes the whole result
        // NULL (handled in the fold); a non-null operand that is neither a string nor a
        // list, or a string/list mix, is a data exception — never JS-string-coerced.
        if name == "concat" {
            let non_null: Vec<&Value> = args.iter().filter(|a| !a.is_null()).collect();
            let all_str = non_null.iter().all(|a| matches!(a, Value::Str(_)));
            let all_list = non_null.iter().all(|a| matches!(a, Value::List(_)));
            if !non_null.is_empty() && !all_str && !all_list {
                return Err(
                    "E_INVALID_VALUE: || requires all operands to be strings or all lists \
                     (values are not coerced)"
                        .into(),
                );
            }
        }
    }
    // `to_boolean(NaN)` / `CAST(NaN AS BOOLEAN)` is an invalid conversion — a NaN is a live
    // value in GQL (only nulled at JSON egress), so it faults at the conversion rather than
    // becoming a null that trips a type error at a later consumer. (`Inf` → true, nonzero.)
    if matches!(name, "to_boolean" | "toboolean")
        && matches!(args.first(), Some(Value::Num(x)) if x.is_nan())
    {
        return Err("E_INVALID_VALUE: cannot convert NaN to a boolean".into());
    }
    Ok(call_scalar(name, args))
}

/// Does the (non-null) value `v` match the IS TYPED scalar category `category`?
/// `integer` requires an integral finite number; `float` any number. Replicates
/// core's `category_matches`/`value_is_typed_ty`.
fn scalar_is_typed(category: &str, v: &Value) -> bool {
    match category {
        "any" => true,
        "null" => false, // v is non-null here
        "bool" => matches!(v, Value::Bool(_)),
        "string" => matches!(v, Value::Str(_)),
        "integer" => matches!(v, Value::Num(n) if n.is_finite() && n.fract() == 0.0),
        "float" => matches!(v, Value::Num(_)),
        "list" => matches!(v, Value::List(_)),
        "record" => matches!(v, Value::Record(_)),
        "date" | "local_time" | "local_datetime" | "zoned_time" | "zoned_datetime" | "duration" => {
            use crate::temporal::TemporalKind as K;
            if let Value::Temporal(t) = v {
                let want = match category {
                    "date" => K::Date,
                    "local_time" => K::Time,
                    "local_datetime" => K::DateTime,
                    "zoned_time" => K::ZonedTime,
                    "zoned_datetime" => K::ZonedDateTime,
                    _ => K::Duration,
                };
                t.kind() == want
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Does the record value `v` conform to a CLOSED record type `schema` (a `Record`
/// of field → descriptor `List[category, not_null, nested_schema_or_null]`)? Closed:
/// every field of `v` must be declared; every declared field must be present-and-
/// typed or absent-and-nullable; a null field value is allowed only when nullable;
/// a `record`-category field recurses into its nested schema (or, for an open
/// `RECORD` with no schema, only checks the value is a record).
fn record_matches_schema(v: &Value, schema: &[(std::sync::Arc<str>, Value)]) -> bool {
    let Value::Record(fields) = v else {
        return false;
    };
    // No undeclared field.
    for (k, _) in fields.iter() {
        if !schema.iter().any(|(sk, _)| sk == k) {
            return false;
        }
    }
    for (sk, desc) in schema.iter() {
        let Value::List(d) = desc else {
            return false;
        };
        let category = match &d[0] {
            Value::Str(s) => s.as_ref(),
            _ => return false,
        };
        let field_not_null = matches!(d[1], Value::Bool(true));
        let nested = &d[2];
        match fields.iter().find(|(fk, _)| fk == sk) {
            None => {
                if field_not_null {
                    return false; // required field absent
                }
            }
            Some((_, fv)) => {
                if fv.is_null() {
                    if field_not_null {
                        return false;
                    }
                } else if category == "record" {
                    match nested {
                        Value::Record(sub) => {
                            if !record_matches_schema(fv, sub) {
                                return false;
                            }
                        }
                        // Open `RECORD` (no `{…}`): only require a record value.
                        _ => {
                            if !matches!(fv, Value::Record(_)) {
                                return false;
                            }
                        }
                    }
                } else if !scalar_is_typed(category, fv) {
                    return false;
                }
            }
        }
    }
    true
}

fn call_scalar(name: &str, args: &[Value]) -> Value {
    match name {
        // variadic
        "coalesce" => args
            .iter()
            .find(|v| !v.is_null())
            .cloned()
            .unwrap_or(Value::Null),
        // `x IS [NOT] TYPED <type> [NOT NULL]` desugars here: args are (value,
        // category, not_null). A NULL value conforms to any nullable type (so it is
        // `!not_null`); else the value's runtime type must match the category —
        // replicated from core's `category_matches`/`value_is_typed_ty`.
        "__is_typed" => {
            let v = &args[0];
            let category = match &args[1] {
                Value::Str(s) => s.as_ref(),
                _ => return Value::Null,
            };
            let not_null = matches!(args[2], Value::Bool(true));
            if v.is_null() {
                return Value::Bool(!not_null);
            }
            Value::Bool(scalar_is_typed(category, v))
        }
        // `x IS [NOT] TYPED RECORD { f :: TYPE [NOT NULL], … }` — a CLOSED record type.
        // arg[1] encodes the schema as a `Record` mapping each field to a descriptor
        // `List[category, not_null, nested_schema_or_null]`. A closed record conforms
        // iff it carries NO undeclared field and every declared field is present-and-
        // typed or absent-and-nullable (recursively for nested records).
        "__is_typed_record" => {
            let v = &args[0];
            let not_null = matches!(args[2], Value::Bool(true));
            if v.is_null() {
                return Value::Bool(!not_null);
            }
            let Value::Record(schema) = &args[1] else {
                return Value::Null;
            };
            Value::Bool(record_matches_schema(v, schema))
        }
        // `a || b || …` — left-associative concat (the parser folds a `||` run into
        // one call). Matches core's `concat_step` fold: ANY null operand → NULL; two
        // lists concatenate element-wise; otherwise both sides JS-string-coerce (via
        // `to_string_fn`) and join.
        "concat" => {
            let mut acc = args.first().cloned().unwrap_or(Value::Null);
            for r in &args[1..] {
                acc = concat_step(&acc, r);
            }
            acc
        }
        // numeric constants (0 args)
        "e" => Value::Num(std::f64::consts::E),
        "pi" => Value::Num(std::f64::consts::PI),
        // numeric (1 arg)
        "abs" | "sign" | "floor" | "ceil" | "ceiling" | "sqrt" | "exp" | "ln" | "log10" | "sin"
        | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "cot"
        | "degrees" | "radians" => scalar_num_fn(name, &args[0]),
        // `round(x)` rounds to an integer; `round(x, digits)` to `digits` decimal
        // places (negative rounds left of the point). Half away from zero, matching
        // core; the `(x*f).round()/f` form is bit-identical (do not reformulate).
        "round" => match value::num_of(&args[0]) {
            Some(x) => {
                let digits = args
                    .get(1)
                    .and_then(value::num_of)
                    .map_or(0, |d| d.trunc() as i32);
                let f = 10f64.powi(digits);
                Value::Num((x * f).round() / f)
            }
            None => Value::Null,
        },
        // numeric (2 args). `log(a, b)` is log-base-a of b = ln(b)/ln(a) (matches
        // core's argument order); `mod` is the fn form of `%` (NaN on a zero
        // divisor — it does NOT throw like the `%` OPERATOR, which core reserves for
        // the operator); `atan2(y, x)` is the two-argument arctangent. NaN/Inf
        // results are KEPT (K4), coerced only at JSON egress.
        "log" | "power" | "mod" | "atan2" => {
            match (value::num_of(&args[0]), value::num_of(&args[1])) {
                (Some(x), Some(y)) => Value::Num(match name {
                    "log" => y.ln() / x.ln(),
                    "power" => x.powf(y),
                    // atan2 is the one math fn whose result distinguishes the sign of a zero
                    // operand (`atan2(-0, -1) = -PI` vs `atan2(+0, -1) = +PI`). We treat -0
                    // and +0 as one value everywhere, so fold -0 to +0 on both inputs (the
                    // pure-TS engine does the same). Every other fn collapses -0 to 0 or null
                    // at egress already.
                    "atan2" => (x + 0.0).atan2(y + 0.0),
                    _ => x % y,
                }),
                _ => Value::Null,
            }
        }
        // nullif(a, b): NULL when a == b (value-contract equality), else a.
        "nullif" => {
            if !args[0].is_null() && !args[1].is_null() && value::equals(&args[0], &args[1]) {
                Value::Null
            } else {
                args[0].clone()
            }
        }
        // Cast FUNCTIONS: NULL on a failed/inapplicable conversion (unlike `CAST`,
        // which throws — and unlike `CAST`, these do NOT coerce a Bool to a number).
        "to_integer" | "tointeger" => to_number(&args[0], true),
        "to_float" | "tofloat" => to_number(&args[0], false),
        "to_string" | "tostring" => to_string_fn(&args[0]),
        "to_boolean" | "toboolean" => to_boolean_fn(&args[0]),
        // `to_list`: a list → itself; a string → its UTF-16 code-unit chars (the JS
        // `split('')` model, kept for byte-identity); a non-nullish scalar → a
        // singleton list; null / non-finite number → null. Matches core's ToList.
        "to_list" | "tolist" => match &args[0] {
            Value::List(_) => args[0].clone(),
            Value::Str(s) => Value::List(
                s.encode_utf16()
                    .map(|u| {
                        Value::Str(std::sync::Arc::from(
                            String::from_utf16_lossy(&[u]).as_str(),
                        ))
                    })
                    .collect(),
            ),
            Value::Num(n) if !n.is_finite() => Value::Null,
            Value::Null => Value::Null,
            other => Value::List(vec![other.clone()]),
        },
        // string (1 arg → string/number)
        "upper" => str_map(&args[0], str::to_uppercase),
        "lower" => str_map(&args[0], str::to_lowercase),
        // `trim` is both-sides; a 2nd (char-set) arg from the SQL-spec form is
        // honored by routing through btrim (identical to core's Trim).
        "trim" => trim_fn("btrim", args),
        // ltrim/rtrim/btrim: 1 arg trims WHITESPACE from that side; a 2nd string
        // arg is the set of characters to strip instead.
        "ltrim" | "rtrim" | "btrim" => trim_fn(name, args),
        // reverse is polymorphic: a string reverses by char, a list by element;
        // anything else is NULL (matches core, e.g. reverse(number) → NULL).
        // reverse: a string reverses by UTF-16 unit (JS model — a surrogate pair
        // reversed decodes lossily to U+FFFD, byte-identical to core), a list by
        // element; anything else is NULL.
        "reverse" => match &args[0] {
            Value::Str(s) => {
                let mut units: Vec<u16> = s.encode_utf16().collect();
                units.reverse();
                Value::Str(String::from_utf16_lossy(&units).into())
            }
            Value::List(v) => Value::List(v.iter().rev().cloned().collect()),
            _ => Value::Null,
        },
        // left/right(s, n): the first / last n UTF-16 units (n ≥ len → the whole
        // string; n ≤ 0 → empty).
        "left" | "right" => match (&args[0], value::num_of(&args[1])) {
            (Value::Str(s), Some(k)) => {
                let units = utf16_len(s);
                let take = (k.max(0.0) as usize).min(units);
                let out = if name == "left" {
                    utf16_slice(s, 0, take)
                } else {
                    utf16_slice(s, units - take, take)
                };
                Value::Str(out.into())
            }
            _ => Value::Null,
        },
        // split(s, delim) → a list of substrings. An EMPTY delimiter splits into one
        // element per UTF-16 unit (JS model), matching core — NOT Rust's `split("")`.
        "split" => match (&args[0], &args[1]) {
            (Value::Str(s), Value::Str(d)) => {
                let parts: Vec<Value> = if d.is_empty() {
                    s.encode_utf16()
                        .map(|u| Value::Str(String::from_utf16_lossy(&[u]).into()))
                        .collect()
                } else {
                    s.split(d.as_ref()).map(|p| Value::Str(p.into())).collect()
                };
                Value::List(parts)
            }
            _ => Value::Null,
        },
        // Length of a string in UTF-16 code units (JS `.length` model), matching
        // core; `byte_length`/`octet_length` count UTF-8 bytes.
        "length" | "char_length" | "character_length" => match &args[0] {
            Value::Str(s) => Value::Num(utf16_len(s) as f64),
            _ => Value::Null,
        },
        "byte_length" | "octet_length" => match &args[0] {
            Value::Str(s) => Value::Num(s.len() as f64),
            _ => Value::Null,
        },
        // string predicates (2 args → bool)
        "starts_with" => str_bool(&args[0], &args[1], |s, sub| s.starts_with(sub)),
        "ends_with" => str_bool(&args[0], &args[1], |s, sub| s.ends_with(sub)),
        "contains" => str_bool(&args[0], &args[1], |s, sub| s.contains(sub)),
        "regex_match" => regex_match(&args[0], &args[1]),
        // replace(s, from[, to]) — `to` defaults to "" (core); an EMPTY search
        // returns the string unchanged (core), NOT Rust's insert-everywhere.
        "replace" => match (&args[0], &args[1]) {
            (Value::Str(s), Value::Str(f)) => {
                let t = match args.get(2) {
                    Some(Value::Str(t)) => t.to_string(),
                    Some(v) if !v.is_null() => return Value::Null,
                    _ => String::new(),
                };
                if f.is_empty() {
                    Value::Str(s.clone())
                } else {
                    Value::Str(s.replace(f.as_ref(), &t).into())
                }
            }
            _ => Value::Null,
        },
        // substring(s, start[, len]) — ISO 1-based, UTF-16-unit indexed
        "substring" => substring(args),
        // `size` (and its ISO/SQL alias `cardinality`) is polymorphic over a collection OR
        // a string (UTF-16 units), like lenke-core; a non-collection non-string is NULL.
        "size" | "cardinality" => match &args[0] {
            Value::List(v) => Value::Num(v.len() as f64),
            Value::Str(s) => Value::Num(utf16_len(s) as f64),
            _ => Value::Null,
        },
        "head" => match &args[0] {
            Value::List(v) => v.first().cloned().unwrap_or(Value::Null),
            _ => Value::Null,
        },
        // tail: all but the first element (empty list → empty).
        "tail" => match &args[0] {
            Value::List(v) => Value::List(v.iter().skip(1).cloned().collect()),
            _ => Value::Null,
        },
        // append(list, x) → the list with x appended.
        "append" => match &args[0] {
            Value::List(v) => {
                let mut out = v.clone();
                out.push(args[1].clone());
                Value::List(out)
            }
            _ => Value::Null,
        },
        // list_contains(list, x) → 1.0 if any element equals x, else 0.0 (a NUMBER,
        // not a bool — matching core; `null` matches `null` via `equals`).
        "list_contains" => match &args[0] {
            Value::List(v) => Value::Num(f64::from(v.iter().any(|e| value::equals(e, &args[1])))),
            _ => Value::Null,
        },
        // list_sort(list, [order], [nullOrder]) — the value contract's total order,
        // reversed for `'desc'`, with absolute null placement (`'first'`/`'last'`,
        // default last). Mirrors ORDER BY / core's compare_sort byte-for-byte. A
        // stored list never holds NaN (it becomes null at ingest), so `is_null`
        // covers every nullish element.
        "list_sort" => {
            match &args[0] {
                Value::List(v) => {
                    let descending = matches!(args.get(1), Some(Value::Str(s)) if s.eq_ignore_ascii_case("desc"));
                    let nulls_first = matches!(args.get(2), Some(Value::Str(s)) if s.eq_ignore_ascii_case("first"));
                    let mut out = v.clone();
                    out.sort_by(|x, y| {
                        use std::cmp::Ordering;
                        match (x.is_null(), y.is_null()) {
                            (true, true) => Ordering::Equal,
                            (true, false) => {
                                if nulls_first {
                                    Ordering::Less
                                } else {
                                    Ordering::Greater
                                }
                            }
                            (false, true) => {
                                if nulls_first {
                                    Ordering::Greater
                                } else {
                                    Ordering::Less
                                }
                            }
                            (false, false) => {
                                let o = value::cmp_total(x, y);
                                if descending {
                                    o.reverse()
                                } else {
                                    o
                                }
                            }
                        }
                    });
                    Value::List(out)
                }
                _ => Value::Null,
            }
        }
        // Set algebra over lists — all DEDUPED (by value equality), matching core.
        // union: a's elements then b's, deduped. intersection: elements of a also
        // in b, deduped. difference: elements of a not in b, deduped.
        "list_union" | "difference" | "intersection" => match (&args[0], &args[1]) {
            (Value::List(a), Value::List(b)) => Value::List(list_set_op(name, a, b)),
            _ => Value::Null,
        },
        // range(start, end[, step]) — INCLUSIVE of both ends; default step 1; a
        // zero step is NULL; a start past end with the wrong sign yields an empty
        // list (matches core).
        "range" => {
            let step = if args.len() == 3 {
                value::as_num(&args[2]).map(f64::trunc)
            } else {
                Some(1.0)
            };
            match (
                value::as_num(&args[0]).map(f64::trunc),
                value::as_num(&args[1]).map(f64::trunc),
                step,
            ) {
                (Some(a), Some(b), Some(st)) if st != 0.0 => {
                    // COUNT-driven, not comparison-driven: `cur += st` stops advancing
                    // once `cur` reaches 2^53 (a no-op in f64), so a `while cur <= b`
                    // loop never terminates even when the count is tiny — e.g.
                    // range(9007199254740992, 9007199254740994) has just 3 elements.
                    // Compute the count up front (matching core), and cap the
                    // allocation. The emitted values still come from repeated addition.
                    let count = ((b - a) / st).floor() + 1.0;
                    if count.is_nan() || count <= 0.0 {
                        Value::List(Vec::new())
                    } else {
                        let n = if count > 10_000_001.0 {
                            10_000_001
                        } else {
                            count as usize
                        };
                        let mut out = Vec::with_capacity(n);
                        let mut cur = a;
                        for _ in 0..n {
                            out.push(Value::Num(cur));
                            cur += st;
                        }
                        Value::List(out)
                    }
                }
                _ => Value::Null,
            }
        }
        "last" => match &args[0] {
            Value::List(v) => v.last().cloned().unwrap_or(Value::Null),
            _ => Value::Null,
        },
        // Temporal component accessors (1 arg → number, or NULL when the component
        // is undefined for that kind). Core spells these with the leading-underscore
        // extension sigil (`_year`); the bare ISO name is not in the grammar.
        "_year" | "_month" | "_day" | "_hour" | "_minute" | "_second" => match &args[0] {
            Value::Temporal(t) => date_part(name.trim_start_matches('_'), *t)
                .map_or(Value::Null, |n| Value::Num(n as f64)),
            _ => Value::Null,
        },
        // Temporal constructors: parse a string, or coerce between kinds.
        "date" => temporal_ctor(&args[0], "date"),
        "local_time" => temporal_ctor(&args[0], "localtime"),
        "datetime" | "local_datetime" => temporal_ctor(&args[0], "datetime"),
        "zoned_time" => temporal_ctor(&args[0], "zoned_time"),
        "zoned_datetime" => temporal_ctor(&args[0], "zoned_datetime"),
        "duration" => temporal_ctor(&args[0], "duration"),
        // The exact span from a to b (b - a), in fixed units; cross-kind → NULL.
        "duration_between" => match (&args[0], &args[1]) {
            (Value::Temporal(x), Value::Temporal(y)) => duration_between(*x, *y),
            _ => Value::Null,
        },
        // Path accessors (nodes/relationships/path_length/elements) are not scalar
        // Call functions — they read the lineage sidecar via `Expr::PathAccess`.
        _ => Value::Null, // parser rejects unknown names; defensive
    }
}

/// Extract a calendar/clock component from a temporal value. `None` when the
/// component is undefined for that kind (`year`/`month`/`day` of a time-only
/// value, or `hour`/`minute`/`second` of a date). Zoned values decompose in their
/// stored offset (the local wall clock), as they render; euclidean division so
/// pre-epoch instants floor correctly. Ported from lenke-core for agreement.
fn date_part(func: &str, t: crate::temporal::Temporal) -> Option<i64> {
    use crate::temporal::{civil_from_days, Temporal};
    const SPD: i64 = 86_400;
    match func {
        "year" | "month" | "day" => {
            let days = match t {
                Temporal::Date(x) => i64::from(x.days),
                Temporal::DateTime(x) => x.secs.div_euclid(SPD),
                Temporal::ZonedDateTime(x) => (x.secs + i64::from(x.offset) * 60).div_euclid(SPD),
                _ => return None,
            };
            let (y, m, d) = civil_from_days(days);
            Some(match func {
                "year" => y,
                "month" => i64::from(m),
                _ => i64::from(d),
            })
        }
        "hour" | "minute" | "second" => {
            let tod = match t {
                Temporal::Time(x) => i64::from(x.secs),
                Temporal::DateTime(x) => x.secs.rem_euclid(SPD),
                Temporal::ZonedTime(x) => {
                    (i64::from(x.secs) + i64::from(x.offset) * 60).rem_euclid(SPD)
                }
                Temporal::ZonedDateTime(x) => (x.secs + i64::from(x.offset) * 60).rem_euclid(SPD),
                _ => return None,
            };
            Some(match func {
                "hour" => tod / 3600,
                "minute" => (tod / 60) % 60,
                _ => tod % 60,
            })
        }
        _ => None,
    }
}

/// Temporal constructor: build a temporal of `kind` from a string (parsed) or
/// coerce another temporal into it (`date(datetime)` → the date part,
/// `datetime(date)` → midnight, `local_time(datetime)` → the time-of-day). A
/// bare `YYYY-MM-DD` string to a datetime target coerces to midnight. Anything
/// with no sensible conversion → NULL. Ported from lenke-core for agreement.
fn temporal_ctor(v: &Value, kind: &str) -> Value {
    use crate::temporal::{Date, DateTime, Temporal, Time};
    const SPD: i64 = 86_400;
    match v {
        // A date-only string to a datetime target → midnight.
        Value::Str(s) if kind == "datetime" && !s.contains(['T', ' ']) => Date::parse(s)
            .map(|d| {
                Value::Temporal(Temporal::DateTime(DateTime {
                    secs: i64::from(d.days) * SPD,
                    nanos: 0,
                }))
            })
            .unwrap_or(Value::Null),
        Value::Str(s) => Temporal::parse(kind, s)
            .map(Value::Temporal)
            .unwrap_or(Value::Null),
        Value::Temporal(t) => match (kind, t) {
            ("date", Temporal::Date(_))
            | ("localtime", Temporal::Time(_))
            | ("datetime", Temporal::DateTime(_))
            | ("duration", Temporal::Duration(_)) => Value::Temporal(*t),
            ("date", Temporal::DateTime(dt)) => Value::Temporal(Temporal::Date(Date {
                days: dt.secs.div_euclid(SPD) as i32,
            })),
            ("localtime", Temporal::DateTime(dt)) => Value::Temporal(Temporal::Time(Time {
                secs: u32::try_from(dt.secs.rem_euclid(SPD)).expect("0..86_400"),
                nanos: dt.nanos,
            })),
            ("datetime", Temporal::Date(d)) => Value::Temporal(Temporal::DateTime(DateTime {
                secs: i64::from(d.days) * SPD,
                nanos: 0,
            })),
            _ => Value::Null, // e.g. duration(date) — no sensible conversion
        },
        _ => Value::Null,
    }
}

/// The EXACT span from `a` to `b` (b − a), in fixed units only: whole days for
/// two dates, seconds+nanos for two datetimes. Any cross-kind pair (or a
/// duration operand) → NULL. Ported from lenke-core.
fn duration_between(a: crate::temporal::Temporal, b: crate::temporal::Temporal) -> Value {
    use crate::temporal::{Duration, Temporal};
    match (a, b) {
        (Temporal::Date(x), Temporal::Date(y)) => Value::Temporal(Temporal::Duration(Duration {
            months: 0,
            days: i64::from(y.days) - i64::from(x.days),
            secs: 0,
            nanos: 0,
        })),
        (Temporal::DateTime(x), Temporal::DateTime(y)) => {
            let mut secs = y.secs - x.secs;
            let mut nanos = i64::from(y.nanos) - i64::from(x.nanos);
            if nanos < 0 {
                nanos += 1_000_000_000;
                secs -= 1;
            }
            Value::Temporal(Temporal::Duration(Duration {
                months: 0,
                days: 0,
                secs,
                nanos: u32::try_from(nanos).expect("0..1e9 after carry"),
            }))
        }
        _ => Value::Null,
    }
}

/// Temporal `+`/`-`/`*` when either operand is temporal: instant ± duration
/// (anchored — months clamped, then days, then time), instant − instant (the
/// exact span), duration ± duration (component-wise), duration × integer. An
/// undefined combination is `Ok(Null)`; a result outside the representable
/// range is a THROWN fault (`Err`) — not a silent null. Ported from lenke-core.
fn temporal_arith(op: crate::ir::ArithOp, lv: &Value, rv: &Value) -> Result<Value, String> {
    use crate::ir::ArithOp::{Add, Mul, Sub};
    use crate::temporal::Temporal as T;
    use Value::Temporal as VT;
    let dur = |r: Option<crate::temporal::Duration>| {
        r.map(|d| VT(T::Duration(d)))
            .ok_or_else(|| "E_INVALID_VALUE: duration component out of range".to_string())
    };
    let inst = |r: Option<T>| {
        r.map(VT)
            .ok_or_else(|| "E_INVALID_VALUE: temporal result out of range".to_string())
    };
    match (op, lv, rv) {
        // duration ± duration (component-wise).
        (Add, VT(T::Duration(a)), VT(T::Duration(b))) => dur(a.add(b)),
        (Sub, VT(T::Duration(a)), VT(T::Duration(b))) => dur(a.add(&b.negate())),
        // instant ± duration (either order for +; dur±dur already handled above).
        (Add, VT(t), VT(T::Duration(d))) | (Add, VT(T::Duration(d)), VT(t)) => {
            inst(t.add_duration(d))
        }
        (Sub, VT(t), VT(T::Duration(d))) => inst(t.add_duration(&d.negate())),
        // instant − instant → the exact span from b to a (a − b).
        (Sub, VT(a), VT(b)) => Ok(duration_between(*b, *a)),
        // duration × INTEGER (either order); a non-integer factor is NULL.
        (Mul, VT(T::Duration(d)), Value::Num(n)) | (Mul, Value::Num(n), VT(T::Duration(d))) => {
            if n.is_finite() && n.fract() == 0.0 {
                dur(d.scale(*n as i64))
            } else {
                Ok(Value::Null)
            }
        }
        _ => Ok(Value::Null),
    }
}

/// Map a string value through `f`; NULL/non-string yields NULL.
fn str_map(v: &Value, f: impl Fn(&str) -> String) -> Value {
    match v {
        Value::Str(s) => Value::Str(f(s).into()),
        _ => Value::Null,
    }
}

/// A two-string predicate; NULL/non-string operand yields NULL.
/// Apply a comparison as a plain bool (UNKNOWN → false). For the shortestPath
/// target filter, where a non-matching/incomparable destination is simply excluded.
fn cmp_apply(op: CompareOp, a: &Value, b: &Value) -> bool {
    match op {
        CompareOp::Eq => value::equals(a, b),
        CompareOp::Ne => !value::equals(a, b),
        CompareOp::Lt => value::cmp_partial(a, b).is_some_and(std::cmp::Ordering::is_lt),
        CompareOp::Le => value::cmp_partial(a, b).is_some_and(std::cmp::Ordering::is_le),
        CompareOp::Gt => value::cmp_partial(a, b).is_some_and(std::cmp::Ordering::is_gt),
        CompareOp::Ge => value::cmp_partial(a, b).is_some_and(std::cmp::Ordering::is_ge),
    }
}

fn str_bool(a: &Value, b: &Value, f: impl Fn(&str, &str) -> bool) -> Value {
    match (a, b) {
        (Value::Str(s), Value::Str(sub)) => Value::Bool(f(s, sub)),
        _ => Value::Null,
    }
}

thread_local! {
    /// Compiled-regex cache for the Gremlin `regex()` predicate, bounded like core's.
    static REGEX_CACHE: std::cell::RefCell<std::collections::HashMap<String, regex::Regex>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// `regex_match(value, pattern)` → Gremlin `regex(pattern)`: true when `value` is a
/// string the pattern finds a match in. A non-string is false; an invalid pattern
/// (already rejected at parse time) is false. Byte-identical to core's `regex_is_match`
/// — same `regex` crate, same bounded thread-local cache.
fn regex_match(v: &Value, pat: &Value) -> Value {
    let (Value::Str(s), Value::Str(p)) = (v, pat) else {
        return Value::Bool(false);
    };
    REGEX_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if !cache.contains_key(p.as_ref()) {
            if cache.len() >= 1000 {
                cache.clear();
            }
            match regex::Regex::new(p) {
                Ok(re) => {
                    cache.insert(p.to_string(), re);
                }
                Err(_) => return Value::Bool(false),
            }
        }
        Value::Bool(cache.get(p.as_ref()).is_some_and(|re| re.is_match(s)))
    })
}

/// Slice `s` by UTF-16 code UNITS `[start, start+len)` (JS `String.slice` /
/// `.length` model), decoding back to UTF-8. A slice that splits a surrogate pair
/// yields U+FFFD there (lossy) — byte-identical to lenke-core (`utf16_slice`) and
/// the TS engine. The whole string model here counts UTF-16 units, NOT `chars()`,
/// so `size('😀')` is 2 (a surrogate pair), matching core.
fn utf16_slice(s: &str, start: usize, len: usize) -> String {
    let units: Vec<u16> = s.encode_utf16().collect();
    let end = start.saturating_add(len).min(units.len());
    let start = start.min(end);
    String::from_utf16_lossy(&units[start..end])
}

/// Length of `s` in UTF-16 code units — the JS `.length` model core uses.
fn utf16_len(s: &str) -> usize {
    s.encode_utf16().count()
}

/// `substring(s, start[, len])` — ISO/SQL **1-based** start, indexed by UTF-16 code
/// UNIT (matching lenke-core exactly). A `start <= 0` shrinks the window from the
/// front (SQL semantics); an omitted `len` runs to the end. NULL for a null string
/// or start.
fn substring(args: &[Value]) -> Value {
    if args[0].is_null() || args[1].is_null() {
        return Value::Null;
    }
    let Value::Str(s) = &args[0] else {
        return Value::Null;
    };
    // 1-based → 0-based offset; a start <= 0 shrinks the window from the front.
    let zero_start = value::num_of(&args[1]).unwrap_or(0.0) - 1.0;
    let from = zero_start.max(0.0) as usize;
    let count = match args.get(2) {
        Some(z) if !z.is_null() => {
            let end = (zero_start + value::num_of(z).unwrap_or(0.0)).max(0.0) as usize;
            end.saturating_sub(from)
        }
        _ => usize::MAX,
    };
    Value::Str(utf16_slice(s, from, count).into())
}

/// Apply a unary numeric scalar function. A NULL / non-numeric argument yields
/// NULL; a computed NaN/Inf result (e.g. `sqrt(-1)`, `ln(0)`) is KEPT (IEEE, like
/// lenke-core — coerced to null only at JSON egress). `sign(0)` is 0 and `sign(NaN)`
/// is NaN (unlike `f64::signum`, which is ±1 for both); rounding is f64's
/// round-half-away-from-zero.
/// The finite→finite unary numeric functions, as raw `f64 -> f64` closures that
/// match [`scalar_num_fn`] EXACTLY. Restricted to functions that cannot introduce
/// NaN/Inf from a finite input (`sqrt`/`ln`/`exp`/… can, so they are excluded):
/// the result column then keeps the all-finite invariant of a stored `Num` column,
/// and the vectorized path is byte-identical to the boxed one. `None` = not eligible.
fn unary_finite_num_fn(name: &str) -> Option<fn(f64) -> f64> {
    Some(match name {
        "abs" => f64::abs,
        "floor" => f64::floor,
        "ceil" | "ceiling" => f64::ceil,
        "round" => f64::round,
        "sign" => |x: f64| {
            if x.is_nan() {
                f64::NAN
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        },
        _ => return None,
    })
}

fn scalar_num_fn(name: &str, v: &Value) -> Value {
    let Some(x) = value::num_of(v) else {
        return Value::Null;
    };
    let r = match name {
        "abs" => x.abs(),
        // NaN is not a number, so it has no sign — `sign(NaN)` stays NaN (→ null at
        // egress), matching JS `Math.sign(NaN)`. Without this guard NaN falls through both
        // `> 0` and `< 0` (both false for NaN) to the `0.0` else-arm, a wrong answer.
        "sign" => {
            if x.is_nan() {
                f64::NAN
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        "floor" => x.floor(),
        "ceil" | "ceiling" => x.ceil(),
        "round" => x.round(),
        "sqrt" => x.sqrt(),
        // Transcendentals — native libm, matching lenke-core's native build. A
        // domain-invalid result (e.g. `ln(-1)`, `cot(0)`) is NaN/Inf and, for now,
        // falls to NULL through the finite gate below (K4 will KEEP it, like core).
        "exp" => x.exp(),
        "ln" => x.ln(),
        "log10" => x.log10(),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "tanh" => x.tanh(),
        "cot" => 1.0 / x.tan(),
        // Multiply-then-divide, NOT `to_degrees`/`to_radians`: the latter pre-round
        // the 180/PI (resp. PI/180) constant and land one ULP off core's byte-exact
        // `(n*180)/PI` / `(n*PI)/180`.
        "degrees" => (x * 180.0) / std::f64::consts::PI,
        "radians" => (x * std::f64::consts::PI) / 180.0,
        _ => return Value::Null, // parser rejects unknown names; defensive
    };
    // NaN/Inf are KEPT (K4) — a computed NaN (`sqrt(-1)`, `ln(-1)`) is a real
    // signal, coerced to null only at the JSON egress boundary, matching core.
    Value::Num(r)
}

/// `to_integer`/`to_float` FUNCTION and the `CAST(x AS INTEGER|FLOAT)` it backs: a Num
/// (truncated for integer), a BOOLEAN (`true`→1, `false`→0 — the ISO-GQL/Ultipa explicit
/// conversion), or a parseable finite string. A list/record/temporal is NULL. (These are
/// EXPLICIT conversions, so a bool converts; the implicit paths — arithmetic, `sum`, … —
/// still never coerce a bool.)
fn to_number(v: &Value, integer: bool) -> Value {
    let n = match v {
        Value::Num(x) => *x,
        Value::Bool(b) => f64::from(u8::from(*b)),
        // A string that parses to a NON-finite value (`'1e1000'` → inf, `'nan'`) is
        // NULL — the fn form never yields inf/NaN, matching core's `.filter(is_finite)`.
        Value::Str(s) => match s.trim().parse::<f64>() {
            Ok(x) if x.is_finite() => x,
            _ => return Value::Null,
        },
        _ => return Value::Null,
    };
    if integer {
        if n.is_finite() {
            Value::Num(n.trunc())
        } else {
            Value::Null
        }
    } else {
        Value::Num(n)
    }
}

/// `to_string` FUNCTION: NULL→NULL, finite Num→its egress text, Bool→"true"/
/// "false", Str→itself, Temporal→its ISO form; a non-finite number is NULL.
/// One step of the `||` fold, matching core's `concat_step`: null propagates, two
/// lists concatenate, otherwise both operands JS-string-coerce and join.
fn concat_step(l: &Value, r: &Value) -> Value {
    if l.is_null() || r.is_null() {
        return Value::Null;
    }
    if let (Value::List(a), Value::List(b)) = (l, r) {
        return Value::List(a.iter().chain(b.iter()).cloned().collect());
    }
    match (to_string_fn(l), to_string_fn(r)) {
        (Value::Str(a), Value::Str(b)) => Value::Str(format!("{a}{b}").into()),
        // A non-stringable operand (e.g. a map) → NULL, as core's js_str-of-unknown does.
        _ => Value::Null,
    }
}

fn to_string_fn(v: &Value) -> Value {
    match v {
        Value::Null => Value::Null,
        Value::Str(s) => Value::Str(s.clone()),
        Value::Bool(b) => Value::Str((if *b { "true" } else { "false" }).into()),
        // Finite number as text, formatted like JS `Number.toString` (`-0` → "0",
        // exponential past the 1e21 / 1e-6 thresholds) — NOT Rust's `{}` (which is decimal
        // at all magnitudes and would give a different STRING for e.g. 1e-7).
        Value::Num(x) if x.is_finite() => Value::Str(crate::json::js_number(*x).into()),
        Value::Temporal(t) => Value::Str(t.format().into()),
        // A LIST joins its elements' string form (a null element → "", like JS
        // `Array.join`); a RECORD/MAP renders as its canonical JSON — matching core's
        // `js_str`, which serializes composites rather than returning NULL.
        Value::List(_) | Value::Record(_) | Value::Map(_) => {
            Value::Str(crate::json::js_str(v).into())
        }
        _ => Value::Null,
    }
}

/// `to_boolean` FUNCTION: a Bool passes through; the strings "true"/"false"
/// (trimmed, case-insensitive) convert; anything else is NULL.
fn to_boolean_fn(v: &Value) -> Value {
    match v {
        Value::Bool(b) => Value::Bool(*b),
        // A number coerces like C truthiness: nonzero → true, zero → false.
        Value::Num(x) => Value::Bool(*x != 0.0),
        Value::Str(s) => {
            let t = s.trim();
            if t.eq_ignore_ascii_case("true") {
                Value::Bool(true)
            } else if t.eq_ignore_ascii_case("false") {
                Value::Bool(false)
            } else {
                Value::Null
            }
        }
        _ => Value::Null,
    }
}

/// Set algebra over two lists, all producing a DEDUPED result (by the value
/// contract's `equals`, so `null` collapses with `null`): `list_union` = a then
/// the b-elements not already present; `intersection` = a-elements also in b;
/// `difference` = a-elements not in b. Order follows first appearance in `a`
/// (then `b` for union). O(n·m) — lists are small.
fn list_set_op(name: &str, a: &[Value], b: &[Value]) -> Vec<Value> {
    let contains = |xs: &[Value], v: &Value| xs.iter().any(|x| value::equals(x, v));
    let mut out: Vec<Value> = Vec::new();
    let push_unique = |out: &mut Vec<Value>, v: &Value| {
        if !contains(out, v) {
            out.push(v.clone());
        }
    };
    match name {
        "intersection" => {
            for v in a {
                if contains(b, v) {
                    push_unique(&mut out, v);
                }
            }
        }
        "difference" => {
            for v in a {
                if !contains(b, v) {
                    push_unique(&mut out, v);
                }
            }
        }
        _ => {
            // union: everything in a, then b's new elements, deduped throughout.
            for v in a.iter().chain(b.iter()) {
                push_unique(&mut out, v);
            }
        }
    }
    out
}

/// `ltrim`/`rtrim`/`btrim`: strip whitespace (1 arg) or a given char set (2 args)
/// from the left / right / both ends of a string. Non-string → NULL.
fn trim_fn(name: &str, args: &[Value]) -> Value {
    let Value::Str(s) = &args[0] else {
        return Value::Null;
    };
    // A 2nd string arg is the set of chars to strip; otherwise strip whitespace.
    let set: Option<Vec<char>> = match args.get(1) {
        None => None,
        Some(Value::Str(cs)) => Some(cs.chars().collect()),
        Some(_) => return Value::Null, // a non-string char set
    };
    let strip = |c: char| {
        set.as_ref()
            .map_or_else(|| c.is_whitespace(), |v| v.contains(&c))
    };
    let trimmed = match name {
        "ltrim" => s.trim_start_matches(strip),
        "rtrim" => s.trim_end_matches(strip),
        _ => s.trim_matches(strip), // btrim
    };
    Value::Str(trimmed.into())
}

/// `op` with its operands swapped — used to normalize `literal <cmp> prop` to
/// `prop <cmp> literal`. Equality is symmetric; the orderings mirror.
fn flip_op(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Eq => CompareOp::Eq,
        CompareOp::Ne => CompareOp::Ne,
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Ge => CompareOp::Le,
    }
}

/// The negation of a compare operator — `NOT (x op y)` ≡ `x invert_op(op) y` for
/// present, finite operands. Stored Num/Str cells always are (NaN/absent → NULL,
/// gated by `present`), so the raw fast paths keep the exact keep-TRUE semantics.
fn invert_op(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Eq => CompareOp::Ne,
        CompareOp::Ne => CompareOp::Eq,
        CompareOp::Lt => CompareOp::Ge,
        CompareOp::Ge => CompareOp::Lt,
        CompareOp::Gt => CompareOp::Le,
        CompareOp::Le => CompareOp::Gt,
    }
}

/// De Morgan push-down of `NOT` for the keep-TRUE filter: an equivalent positive
/// predicate (`NOT` eliminated) when `e` is built from compares / AND / OR / NOT,
/// else `None`. Exact in Kleene 3-valued logic for "keep rows where TRUE": `NOT e`
/// is TRUE iff `e` is FALSE, and each rule preserves that (`AND` is FALSE iff an
/// operand is FALSE → `OR` of the negations; `>` inverts to `<=`; etc.). Absent /
/// NaN cells stay dropped on both sides because every compare is UNKNOWN there.
fn invert_pred(e: &Expr) -> Option<Expr> {
    Some(match e {
        Expr::Compare { op, left, right } => Expr::Compare {
            op: invert_op(*op),
            left: left.clone(),
            right: right.clone(),
        },
        Expr::And(a, b) => Expr::Or(Box::new(invert_pred(a)?), Box::new(invert_pred(b)?)),
        Expr::Or(a, b) => Expr::And(Box::new(invert_pred(a)?), Box::new(invert_pred(b)?)),
        Expr::Not(inner) => (**inner).clone(),
        _ => return None,
    })
}

/// A typed reader over ONE storage column for the multi-column distinct fast path:
/// it appends a row's grouping-key bytes (byte-identical to
/// [`value::group_key_into`] over the boxed value, so the induced equivalence is the
/// same) and produces the row's output `Value` — both reading the column directly,
/// borrowing a `&str` for the key rather than boxing or cloning per row. A `Dict`
/// column keys on its decoded string, exactly as a `Str` would.
/// One column's contribution to a composite DISTINCT key — the typed alternative to the
/// byte-key, so a high-card Str cell hashes as a BORROWED `&str` (no per-node byte copy). The
/// key tuple is positional, so parts of different types never need a discriminating tag.
#[derive(Clone, PartialEq, Eq, Hash)]
enum KeyPart<'a> {
    Absent,
    Bool(u8),
    Num(u64),
    Code(u32),
    Str(&'a str),
}

enum ColKeyer<'a> {
    Dict {
        dict: &'a [std::sync::Arc<str>],
        codes: &'a [u32],
        present: &'a [bool],
    },
    Num {
        data: &'a [f64],
        present: &'a [bool],
    },
    Str {
        data: &'a [std::sync::Arc<str>],
        present: &'a [bool],
    },
    Bool {
        data: &'a [bool],
        present: &'a [bool],
    },
}

impl<'a> ColKeyer<'a> {
    /// A keyer for a Num/Str/Bool/Dict column; `None` for Temporal/Gen/missing (which
    /// may carry present-null or need typed compare — left to the general path).
    fn of(col: Option<&'a Column>) -> Option<Self> {
        match col? {
            Column::Dict {
                dict,
                codes,
                present,
                ..
            } => Some(Self::Dict {
                dict,
                codes,
                present,
            }),
            Column::Num { data, present, .. } => Some(Self::Num { data, present }),
            Column::Str { data, present, .. } => Some(Self::Str { data, present }),
            Column::Bool { data, present, .. } => Some(Self::Bool { data, present }),
            _ => None,
        }
    }

    /// Append row `i`'s grouping-key bytes. Str/Num/Bool mirror `group_key_into`
    /// tag-for-tag (absent → `0`, bool → `1`, num → `2`, str → `3`). A `Dict` column
    /// instead keys on its `u32` CODE (tag `8`): the dict assigns exactly one code
    /// per distinct string, so two rows share a code iff they share the string —
    /// the same equivalence a string key induces, but 4 bytes and no string hash.
    /// Codes never cross columns (each column keys at its own fixed offset).
    fn key_into(&self, i: usize, out: &mut Vec<u8>) {
        let push_str = |out: &mut Vec<u8>, s: &str| {
            out.push(3);
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        };
        match self {
            Self::Dict { codes, present, .. } => {
                if present[i] {
                    out.push(8);
                    out.extend_from_slice(&codes[i].to_le_bytes());
                } else {
                    out.push(0);
                }
            }
            Self::Str { data, present } => {
                if present[i] {
                    push_str(out, &data[i]);
                } else {
                    out.push(0);
                }
            }
            Self::Num { data, present } => {
                if present[i] {
                    out.push(2);
                    out.extend_from_slice(&value::num_group_bits(data[i]).to_le_bytes());
                } else {
                    out.push(0);
                }
            }
            Self::Bool { data, present } => {
                if present[i] {
                    out.push(1);
                    out.push(u8::from(data[i]));
                } else {
                    out.push(0);
                }
            }
        }
    }

    /// Row `i`'s composite-key PART — the typed value the byte-key encodes, but BORROWING a
    /// Str cell's `&str` instead of copying its bytes (a high-card Str column is where the
    /// byte-key's per-node alloc+copy dominates; the borrow hashes the same content with no
    /// copy). Positional in the key tuple, so no cross-column tag is needed: a column is a
    /// fixed type, and `Dict` keys on its CODE exactly as the byte-key does (same string →
    /// same code). `Num` uses `num_group_bits` so the induced equivalence matches the byte-key
    /// (−0.0/0.0 and the NaNs collapse identically).
    fn key_part(&self, i: usize) -> KeyPart<'a> {
        match self {
            Self::Dict { codes, present, .. } => {
                if present[i] {
                    KeyPart::Code(codes[i])
                } else {
                    KeyPart::Absent
                }
            }
            Self::Str { data, present } => {
                if present[i] {
                    KeyPart::Str(&data[i])
                } else {
                    KeyPart::Absent
                }
            }
            Self::Num { data, present } => {
                if present[i] {
                    KeyPart::Num(value::num_group_bits(data[i]))
                } else {
                    KeyPart::Absent
                }
            }
            Self::Bool { data, present } => {
                if present[i] {
                    KeyPart::Bool(u8::from(data[i]))
                } else {
                    KeyPart::Absent
                }
            }
        }
    }

    /// Row `i`'s output value (absent → `Null`). Clones an `Arc` only here — called
    /// once per SURVIVING distinct tuple, not per scanned row.
    fn value_at(&self, i: usize) -> Value {
        match self {
            Self::Dict {
                dict,
                codes,
                present,
            } => {
                if present[i] {
                    Value::Str(dict[codes[i] as usize].clone())
                } else {
                    Value::Null
                }
            }
            Self::Str { data, present } => {
                if present[i] {
                    Value::Str(data[i].clone())
                } else {
                    Value::Null
                }
            }
            Self::Num { data, present } => {
                if present[i] {
                    Value::Num(data[i])
                } else {
                    Value::Null
                }
            }
            Self::Bool { data, present } => {
                if present[i] {
                    Value::Bool(data[i])
                } else {
                    Value::Null
                }
            }
        }
    }
}

/// Fused multi-column `RETURN DISTINCT n.a, n.b, …` over a bare `Scan`: read the
/// storage columns directly and dedup on a composite grouping key, emitting only the
/// distinct tuples (first-seen order) — so the 100k-row projected columns (a `dept`
/// of `Arc<str>` above all) are never materialized and no `Value` is boxed per
/// scanned row. `None` unless the input is a `Project(Scan, [prop, …])` whose every
/// key is a plain (non-dotted) property backed by a Num/Str/Bool/Dict column.
fn try_distinct_scan_multi(input: &Plan, store: &Store) -> Option<Batch> {
    let Plan::Project { input: scan, items } = input else {
        return None;
    };
    let Plan::Scan { label } = scan.as_ref() else {
        return None;
    };
    if items.is_empty() {
        return None;
    }
    let mut readers: Vec<ColKeyer> = Vec::with_capacity(items.len());
    for (_, e) in items {
        let Expr::Prop { slot: 0, key } = e else {
            return None;
        };
        if key.contains('.') {
            return None; // a dotted record path — leave to the general path
        }
        readers.push(ColKeyer::of(store.column(key))?);
    }

    let ncol = readers.len();
    let mut outs: Vec<Vec<Value>> = vec![Vec::new(); ncol];
    let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
    let mut buf: Vec<u8> = Vec::new();
    scan_visit(store, label, |i| {
        buf.clear();
        for r in &readers {
            r.key_into(i, &mut buf);
        }
        if !seen.contains(buf.as_slice()) {
            seen.insert(buf.clone());
            for (c, r) in readers.iter().enumerate() {
                outs[c].push(r.value_at(i));
            }
        }
    });
    Some(Batch::of(outs.into_iter().map(Col::Gen).collect()))
}

/// Dedup a materialized batch's whole rows: the typed single-column fast path (raw value,
/// no byte-key) else a per-row composite group-key. First-seen order.
fn distinct_batch(batch: Batch) -> Batch {
    if let Some(keep) = try_distinct_typed(&batch) {
        return batch.gather(&keep);
    }
    let n = batch.rows();
    let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
    let mut buf = Vec::new();
    let keep: Vec<usize> = (0..n)
        .filter(|&i| {
            buf.clear();
            for c in &batch.slots {
                value::group_key_into(&c.value_at(i), &mut buf);
            }
            if seen.contains(buf.as_slice()) {
                false
            } else {
                seen.insert(buf.clone());
                true
            }
        })
        .collect();
    batch.gather(&keep)
}

/// `DISTINCT <expr(endpoint)…>` over a (optionally endpoint-WHERE'd) var-length hop, where
/// every projected expression reads ONLY the endpoint slot. Dedup the reachable endpoints
/// (no path materialization), evaluate the projection over just them, then dedup the
/// projected rows — byte-identical to materialize → project → dedup (the projection depends
/// only on the endpoint, so deduping endpoints first can't change the distinct result or its
/// first-seen order). `None` when not that shape (caller falls back). Bare-Prop projections
/// are already handled by `try_distinct_frontier_prop`/`_multi`; this catches expressions.
fn try_distinct_varlen_expr(input: &Plan, store: &Store) -> Option<Batch> {
    let Plan::Project {
        input: chain,
        items,
    } = input
    else {
        return None;
    };
    let endpoint = chain_pull_width(chain)?.checked_sub(1)?;
    if !items.iter().all(|(_, e)| refs_only_slot(e, endpoint)) {
        return None;
    }
    if !chain_has_varlen(chain) {
        return None; // fixed chains keep the existing frontier/materialize path
    }
    let eps = distinct_chain_endpoints(chain, store)?;
    let n = eps.len();
    let mut cols: Vec<Col> = (0..endpoint).map(|_| Col::Nodes(vec![0u32; n])).collect();
    cols.push(Col::Nodes(eps));
    let batch = Batch::of(cols);
    let projected = eval_all(items.iter().map(|(_, e)| e), store, &batch).ok()?;
    Some(distinct_batch(Batch::of(projected)))
}

/// Does the chain contain a var-length hop? Only then is the endpoint-dedup worth its
/// per-node bitsets over the materialize path (a pure fixed chain has no path explosion).
fn chain_has_varlen(p: &Plan) -> bool {
    match p {
        Plan::VarLength { .. } => true,
        Plan::Expand { input, .. } | Plan::Filter { input, .. } => chain_has_varlen(input),
        _ => false,
    }
}

/// The DISTINCT reachable-endpoint SET of a chain, deduping at EVERY hop instead of
/// materializing paths — the reachable set is all a DISTINCT (or `min`/`max`) over the
/// endpoint depends on, so this is byte-identical to materialize-then-dedup (an unordered
/// result is set-compared). O(V+E), not O(paths): a var-length hop runs the dedup sink; a
/// fixed hop takes the deduped neighbours of the deduped source; a Filter narrows by an
/// endpoint-only predicate. `None` (caller falls back) for a branch/bound-edge/re-entrant
/// chain, an indexed bare-equality WHERE (better served by the reverse seed), or a
/// predicate that reads a non-endpoint slot.
fn distinct_chain_endpoints(chain: &Plan, store: &Store) -> Option<Vec<u32>> {
    let n = store.node_count();
    match chain {
        Plan::Scan { .. } | Plan::IndexSeek { .. } | Plan::RangeSeek { .. } => {
            let mut f = frontier_ids(chain, store)?;
            let mut seen = vec![false; n];
            f.retain(|&x| x != u32::MAX && !std::mem::replace(&mut seen[x as usize], true));
            Some(f)
        }
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            bind_edge: false,
            double_loops: false,
        } => {
            if *from != chain_pull_width(input)?.checked_sub(1)? {
                return None; // must expand from the current endpoint (a straight chain)
            }
            let src = distinct_chain_endpoints(input, store)?;
            let want = want_etypes(store, edge_label).ok()?;
            let mut seen = vec![false; n];
            let mut out = Vec::new();
            for &s in &src {
                for_each_nbr(store, s, *dir, &want, false, |nbr, _| {
                    if !std::mem::replace(&mut seen[nbr as usize], true) {
                        out.push(nbr);
                    }
                });
            }
            Some(out)
        }
        Plan::VarLength {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            mode,
            until: None,
            body_filter: None,
            double_loops,
        } => {
            if *from != chain_pull_width(input)?.checked_sub(1)? {
                return None;
            }
            let src = distinct_chain_endpoints(input, store)?;
            let want = match want_etypes(store, edge_label) {
                Ok(w) => w,
                Err(()) => vec![u32::MAX],
            };
            let mut sink = DistinctEndpointEmit {
                seen: vec![false; n],
                out: Vec::new(),
            };
            run_varlen(
                &src,
                store,
                &want,
                *min,
                *max,
                *dir,
                *mode,
                None,
                1,
                None,
                None,
                *double_loops,
                &mut sink,
            );
            Some(sink.out)
        }
        Plan::Filter { input, pred } => {
            let endpoint = chain_pull_width(input)?.checked_sub(1)?;
            // An indexed bare-equality endpoint is better served by the reverse seed (seed
            // the ~1 node) — decline. Only an endpoint-only predicate can filter the set.
            if let Some((k, _)) = target_eq(pred, endpoint) {
                if store.has_hash_index(&k) {
                    return None;
                }
            }
            if !refs_only_slot(pred, endpoint) {
                return None;
            }
            let eps = distinct_chain_endpoints(input, store)?;
            let rows = eps.len();
            let mut cols: Vec<Col> = (0..endpoint)
                .map(|_| Col::Nodes(vec![0u32; rows]))
                .collect();
            cols.push(Col::Nodes(eps));
            let batch = Batch::of(cols);
            let mask = eval_mask(pred, store, &batch).ok()?;
            let Col::Nodes(eps) = batch.slot(endpoint) else {
                return None;
            };
            Some(
                eps.iter()
                    .enumerate()
                    .filter(|&(i, _)| mask.get(i) == Some(&Some(true)))
                    .map(|(_, &e)| e)
                    .collect(),
            )
        }
        _ => None,
    }
}

/// The row-order endpoint frontier of a chain for a fused DISTINCT path: a pure Scan/Expand
/// chain yields just the endpoint ids directly (`frontier_ids` — no intermediate slots
/// materialized, unlike a full `pull`); a filtered chain is pulled once and its endpoint slot
/// cloned out.
fn chain_frontier(chain: &Plan, store: &Store, endpoint: usize) -> Option<Vec<u32>> {
    // A chain containing a var-length: DISTINCT needs only the reachable-endpoint SET, so
    // dedup at every hop instead of materializing every path. (A fixed-hop fan-out is left
    // to frontier_ids + the caller's node-dedup bitset — routing it through the recursion
    // here only adds a redundant second bitset for no measured gain.)
    if chain_has_varlen(chain) {
        if let Some(eps) = distinct_chain_endpoints(chain, store) {
            return Some(eps);
        }
    }
    match frontier_ids(chain, store) {
        Some(f) => Some(f),
        None => {
            let b = pull(chain, store, false).ok()?;
            match b.slot(endpoint) {
                Col::Nodes(f) => Some(f.clone()),
                _ => None,
            }
        }
    }
}

/// The frontier sibling of [`try_distinct_scan_prop`]: single-column `RETURN DISTINCT b.k`
/// where `b` is a HOP-CHAIN endpoint. Pull the chain (cheap `Col::Nodes`, no property
/// column), then dedup the endpoint's values off storage with a TYPED set — `FnvSet<&str>`
/// for Str, a per-code bitset for Dict, `FnvSet<u64>` (group bits) for Num — instead of the
/// composite byte-key. This is the single-column case the multi-column
/// [`try_distinct_frontier_multi`] deliberately bails on (a raw Str/Num loses to the typed
/// set there). Absence is one `Null` row (first-seen); DISTINCT order is set-compared.
fn try_distinct_frontier_prop(input: &Plan, store: &Store) -> Option<Batch> {
    let Plan::Project {
        input: chain,
        items,
    } = input
    else {
        return None;
    };
    let [(_, Expr::Prop { slot, key })] = items.as_slice() else {
        return None;
    };
    if key.contains('.') {
        return None;
    }
    let endpoint = chain_pull_width(chain)?.checked_sub(1)?;
    if *slot != endpoint {
        return None;
    }
    let col = store.column(key)?;
    if !matches!(
        col,
        Column::Str { .. } | Column::Dict { .. } | Column::Num { .. } | Column::Bool { .. }
    ) {
        return None; // Temporal / Gen → the general path
    }
    let frontier = chain_frontier(chain, store, endpoint)?;
    let frontier: &[u32] = &frontier;
    let mut out: Vec<Value> = Vec::new();
    let mut saw_null = false;
    let null_once = |out: &mut Vec<Value>, saw: &mut bool| {
        if !*saw {
            *saw = true;
            out.push(Value::Null);
        }
    };
    match col {
        Column::Str { data, present, .. } => {
            // A hop endpoint repeats (degree-many paths reach it), and string hashing is
            // the cost — so dedup the NODES first with a cheap bitset and hash only each
            // distinct node's string once. Order is unchanged: a node's first occurrence
            // still drives insertion, later ones are skipped (before they were re-hashed
            // and dropped by the string set). Different nodes with equal strings still
            // collapse via the string set.
            let mut seen_node = vec![false; store.node_count()];
            let mut seen: FnvSet<&str> = FnvSet::default();
            for &node in frontier {
                if node != u32::MAX && present[node as usize] {
                    let i = node as usize;
                    if std::mem::replace(&mut seen_node[i], true) {
                        continue; // duplicate endpoint node — already accounted for
                    }
                    if seen.insert(data[i].as_ref()) {
                        out.push(Value::Str(data[i].clone()));
                    }
                } else {
                    null_once(&mut out, &mut saw_null);
                }
            }
        }
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            let mut seen = vec![false; dict.len()];
            for &node in frontier {
                if node != u32::MAX && present[node as usize] {
                    let c = codes[node as usize] as usize;
                    if !std::mem::replace(&mut seen[c], true) {
                        out.push(Value::Str(dict[c].clone()));
                    }
                } else {
                    null_once(&mut out, &mut saw_null);
                }
            }
        }
        Column::Num { data, present, .. } => {
            let mut seen: FnvSet<u64> = FnvSet::default();
            for &node in frontier {
                if node != u32::MAX && present[node as usize] {
                    let i = node as usize;
                    if seen.insert(value::num_group_bits(data[i])) {
                        out.push(Value::Num(data[i]));
                    }
                } else {
                    null_once(&mut out, &mut saw_null);
                }
            }
        }
        Column::Bool { data, present, .. } => {
            let mut seen = [false; 2];
            for &node in frontier {
                if node != u32::MAX && present[node as usize] {
                    let b = usize::from(data[node as usize]);
                    if !std::mem::replace(&mut seen[b], true) {
                        out.push(Value::Bool(data[node as usize]));
                    }
                } else {
                    null_once(&mut out, &mut saw_null);
                }
            }
        }
        _ => return None,
    }
    Some(Batch::single(Col::Gen(out)))
}

/// The frontier sibling of [`try_distinct_scan_multi`]: `RETURN DISTINCT b.a, b.b, …`
/// where `b` is a HOP-CHAIN endpoint. Pull the chain (cheap `Col::Nodes` columns — the
/// traversal is unavoidable, but NO per-hop property column is built), then key each
/// endpoint node straight off storage via [`ColKeyer`] (a 4-byte dict CODE, not a hashed
/// string) and clone an `Arc` only for a surviving tuple. This drops the two costs the
/// general path pays over the exploded frontier: materializing full `Arc<str>` property
/// columns (`eval_all`) and byte-keying decoded strings. Dedup is first-seen over the
/// batch's row order — the same order the general dedup sees — so it is byte-identical.
/// `None` unless every projected key is a plain property of the chain frontier backed by a
/// Num/Str/Bool/Dict column.
/// `RETURN DISTINCT x, x, …, x` where every projection item is the SAME property: the
/// distinct tuples are `{(v, …, v) : v ∈ distinct(x)}` in `x`'s first-seen order, i.e. the
/// single-column DISTINCT with its output column replicated. Route to the fast single-column
/// path (typed set / dict-code bitset) and clone the one result column, instead of the
/// composite byte-key that keys+clones the identical column N times. `None` unless the input
/// is a Project whose ≥2 items are all the identical `Prop` (the `b.city, b.city` shape the
/// fuzzer emits). Lineage-free (the caller gates on `!track`; DISTINCT collapses paths).
fn try_distinct_identical_cols(input: &Plan, store: &Store, track: bool) -> Option<Batch> {
    let Plan::Project {
        input: chain,
        items,
    } = input
    else {
        return None;
    };
    if items.len() < 2 {
        return None;
    }
    let Expr::Prop { slot: s0, key: k0 } = &items[0].1 else {
        return None;
    };
    if !items
        .iter()
        .all(|(_, e)| matches!(e, Expr::Prop { slot, key } if slot == s0 && key == k0))
    {
        return None;
    }
    let single = Plan::Distinct {
        input: Box::new(Plan::Project {
            input: chain.clone(),
            items: vec![items[0].clone()],
        }),
    };
    let b = pull(&single, store, track).ok()?;
    let col = b.slots.into_iter().next()?;
    Some(Batch::of(vec![col; items.len()]))
}

fn try_distinct_frontier_multi(input: &Plan, store: &Store) -> Option<Batch> {
    let Plan::Project {
        input: chain,
        items,
    } = input
    else {
        return None;
    };
    if items.is_empty() {
        return None;
    }
    let endpoint = chain_pull_width(chain)?.checked_sub(1)?;
    let mut readers: Vec<ColKeyer> = Vec::with_capacity(items.len());
    for (_, e) in items {
        let Expr::Prop { slot, key } = e else {
            return None;
        };
        if *slot != endpoint || key.contains('.') {
            return None; // must be a plain property of the chain frontier
        }
        readers.push(ColKeyer::of(store.column(key))?);
    }
    // A single non-DICT column is better served by the typed dedup after the (cheap) pull:
    // for a raw Str/Num the byte-key here loses to its `FnvSet<&str>` / f64-bits set. The
    // byte-key only pays off when it skips a Dict DECODE (a low-card code, 4 bytes) or when
    // a composite (multi-column) key is unavoidable anyway.
    if readers.len() == 1 && !matches!(readers[0], ColKeyer::Dict { .. }) {
        return None;
    }
    // A pure Scan/Expand chain yields just the endpoint ids (no intermediate slots
    // materialized); a filtered chain is pulled once and its endpoint extracted.
    let frontier = chain_frontier(chain, store, endpoint)?;
    let frontier: &[u32] = &frontier;
    let ncol = readers.len();
    let mut outs: Vec<Vec<Value>> = vec![Vec::new(); ncol];
    // A hop endpoint repeats; building+hashing the composite key is the cost, so skip duplicate
    // NODES with a cheap bitset and key only each distinct node once. Order-preserving (first
    // occurrence drives insertion); `u32::MAX` (optional-unmatched) reads as all-Absent/all-NULL,
    // so it dedups against an all-absent real node identically.
    let mut seen_node = vec![false; store.node_count()];
    // TWO-column fast path WITH a high-card Str column: a FIXED `(KeyPart, KeyPart)` tuple — no
    // per-node heap alloc, and it BORROWS the Str cell's `&str` (no byte copy). That copy+alloc is
    // what makes a Str composite lose; for all-Num / Dict pairs the compact byte-key is smaller
    // than the enum tuple, so they keep it.
    if ncol == 2 && readers.iter().any(|r| matches!(r, ColKeyer::Str { .. })) {
        let (r0, r1) = (&readers[0], &readers[1]);
        let mut seen: FnvSet<(KeyPart, KeyPart)> = FnvSet::default();
        for &node in frontier {
            if node != u32::MAX && std::mem::replace(&mut seen_node[node as usize], true) {
                continue;
            }
            let key = if node == u32::MAX {
                (KeyPart::Absent, KeyPart::Absent)
            } else {
                let i = node as usize;
                (r0.key_part(i), r1.key_part(i))
            };
            if seen.insert(key) {
                for (c, r) in readers.iter().enumerate() {
                    outs[c].push(if node == u32::MAX {
                        Value::Null
                    } else {
                        r.value_at(node as usize)
                    });
                }
            }
        }
        return Some(Batch::of(outs.into_iter().map(Col::Gen).collect()));
    }
    // General N-column path: a byte-key tuple (`u32::MAX` → one all-NULL key per column).
    let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
    let mut buf: Vec<u8> = Vec::new();
    for &node in frontier {
        if node != u32::MAX && std::mem::replace(&mut seen_node[node as usize], true) {
            continue;
        }
        buf.clear();
        if node == u32::MAX {
            buf.resize(readers.len(), 0);
        } else {
            for r in &readers {
                r.key_into(node as usize, &mut buf);
            }
        }
        if !seen.contains(buf.as_slice()) {
            seen.insert(buf.clone());
            for (c, r) in readers.iter().enumerate() {
                outs[c].push(if node == u32::MAX {
                    Value::Null
                } else {
                    r.value_at(node as usize)
                });
            }
        }
    }
    Some(Batch::of(outs.into_iter().map(Col::Gen).collect()))
}

/// One-pass predicate for the common `<prop> <cmp> <literal>` (either operand
/// order) over a node frontier: read the storage property per row and emit the
/// kept row indices, without building a full value column AND a full boolean mask
/// as intermediates. Every comparison goes through the value contract, so results
/// match the general path exactly: an absent property is NULL → UNKNOWN → dropped,
/// a NULL literal makes every comparison UNKNOWN → all dropped, and cross-type is
/// the contract's `equals`/`cmp_total`. `None` if the predicate is not this shape.
/// Fused `RETURN DISTINCT n.k` — a `Distinct` over a `Project(Scan, [one prop])` —
/// reading the storage column directly and deduping to just the distinct values
/// (first-seen order), so the 100k-row projected column is never materialized.
/// Absence is a distinct value (a present-null / missing prop → one `Null` row, as
/// grouping treats it). `None` unless the shape is exactly that over a `Num`/`Str`/
/// `Bool` column.
fn try_distinct_scan_prop(input: &Plan, store: &Store) -> Option<Batch> {
    let Plan::Project { input: scan, items } = input else {
        return None;
    };
    let [(_, Expr::Prop { slot: 0, key })] = items.as_slice() else {
        return None;
    };
    let Plan::Scan { label } = scan.as_ref() else {
        return None;
    };
    let mut out: Vec<Value> = Vec::new();
    let mut saw_null = false;
    match store.column(key)? {
        Column::Str { data, present, .. } => {
            let mut seen: FnvSet<&str> = FnvSet::default();
            scan_visit(store, label, |i| {
                if present[i] {
                    if seen.insert(data[i].as_ref()) {
                        out.push(Value::Str(data[i].clone()));
                    }
                } else if !saw_null {
                    saw_null = true;
                    out.push(Value::Null);
                }
            });
        }
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            // First-seen order is preserved by pushing when a code is first observed
            // during the scan (NOT dict order, which can differ from scan order under
            // deletes / a label subset).
            let mut seen = vec![false; dict.len()];
            scan_visit(store, label, |i| {
                if present[i] {
                    let c = codes[i] as usize;
                    if !std::mem::replace(&mut seen[c], true) {
                        out.push(Value::Str(dict[c].clone()));
                    }
                } else if !saw_null {
                    saw_null = true;
                    out.push(Value::Null);
                }
            });
        }
        Column::Num { data, present, .. } => {
            // Low-card integer fast path: recover the distinct values from a bitset
            // (ascending) instead of hashing every cell. DISTINCT output order is
            // unspecified (compared as a set), so ascending is fine; a NULL is still
            // emitted once if any cell is absent.
            if let Some((lo, bits, saw_absent)) = low_card_int_bitset(store, label, data, present) {
                if saw_absent {
                    out.push(Value::Null);
                }
                for (k, &set) in bits.iter().enumerate() {
                    if set {
                        out.push(Value::Num(lo + k as f64));
                    }
                }
            } else {
                let mut seen: FnvSet<u64> = FnvSet::default();
                scan_visit(store, label, |i| {
                    if present[i] {
                        if seen.insert(value::num_group_bits(data[i])) {
                            out.push(Value::Num(data[i]));
                        }
                    } else if !saw_null {
                        saw_null = true;
                        out.push(Value::Null);
                    }
                });
            }
        }
        Column::Bool { data, present, .. } => {
            let mut seen = [false; 2];
            scan_visit(store, label, |i| {
                if present[i] {
                    let b = data[i];
                    if !std::mem::replace(&mut seen[usize::from(b)], true) {
                        out.push(Value::Bool(b));
                    }
                } else if !saw_null {
                    saw_null = true;
                    out.push(Value::Null);
                }
            });
        }
        _ => return None, // Temporal / Gen → the general Distinct path
    }
    Some(Batch::of(vec![Col::Gen(out)]))
}

/// Row indices of the first occurrence of each distinct value in a SINGLE-column
/// batch, keyed by the raw value (`&str`, f64 group bits, or a dense id) rather
/// than a serialized byte key — the common `RETURN DISTINCT n.k` shape. `None` for
/// a multi-column batch or a `Gen` column (which may hold nulls/mixed types, where
/// the grouping-byte key is needed). First-seen order preserved.
fn try_distinct_typed(batch: &Batch) -> Option<Vec<usize>> {
    let [col] = batch.slots.as_slice() else {
        return None;
    };
    let mut keep = Vec::new();
    match col {
        Col::Str(v) => {
            let mut seen: FnvSet<&str> = FnvSet::default();
            for (i, s) in v.iter().enumerate() {
                if seen.insert(s.as_ref()) {
                    keep.push(i);
                }
            }
        }
        Col::Num(v) => {
            // f64 group bits collapse NaN payloads and signed zero, matching the
            // grouping contract.
            let mut seen: FnvSet<u64> = FnvSet::default();
            for (i, &x) in v.iter().enumerate() {
                if seen.insert(value::num_group_bits(x)) {
                    keep.push(i);
                }
            }
        }
        Col::Nodes(v) | Col::Edges(v) => {
            let mut seen: FnvSet<u32> = FnvSet::default();
            for (i, &id) in v.iter().enumerate() {
                if seen.insert(id) {
                    keep.push(i);
                }
            }
        }
        Col::Bool(v) => {
            let mut seen = [false; 2];
            for (i, &b) in v.iter().enumerate() {
                if !std::mem::replace(&mut seen[usize::from(b)], true) {
                    keep.push(i);
                }
            }
        }
        Col::Gen(_) => return None, // nulls / mixed types → the grouping-byte key
    }
    Some(keep)
}

/// Flatten a conjunction into its atoms (`a AND b AND c` → `[a, b, c]`); a
/// non-`And` expression is a single atom.
fn flatten_and<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
        Expr::And(a, b) => {
            flatten_and(a, out);
            flatten_and(b, out);
        }
        _ => out.push(e),
    }
}

fn flatten_or<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
        Expr::Or(a, b) => {
            flatten_or(a, out);
            flatten_or(b, out);
        }
        _ => out.push(e),
    }
}

/// Keep rows satisfying a DISJUNCTION of `prop <op> num-literal` compares, all on
/// the same node slot and all reading `Num` columns — one raw-f64 pass keeping a
/// row when ANY disjunct is TRUE. This is the OR mirror of [`try_num_conjunction`],
/// and it also catches `x IN [a, b, …]`, which the parser desugars to an OR-chain of
/// equalities. 3VL WHERE semantics hold: a disjunct over a NULL/NaN cell is never
/// TRUE, so the row is kept iff some disjunct is definitely TRUE (else FALSE/UNKNOWN
/// → dropped), matching the general `Or` evaluator under `is_true`.
fn try_num_disjunction(pred: &Expr, store: &Store, batch: &Batch) -> Option<Vec<usize>> {
    if !matches!(pred, Expr::Or(..)) {
        return None;
    }
    let mut atoms = Vec::new();
    flatten_or(pred, &mut atoms);
    let mut slot0: Option<usize> = None;
    let mut specs: Vec<(&[f64], &[bool], CompareOp, f64)> = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let Expr::Compare { op, left, right } = atom else {
            return None;
        };
        let (slot, key, op, lit) = match (left.as_ref(), right.as_ref()) {
            (Expr::Prop { slot, key }, Expr::Lit(v)) => (*slot, key, *op, v),
            (Expr::Lit(v), Expr::Prop { slot, key }) => (*slot, key, flip_op(*op), v),
            _ => return None,
        };
        match slot0 {
            Some(s) if s != slot => return None, // all disjuncts on the same slot
            _ => slot0 = Some(slot),
        }
        let Value::Num(t) = lit else { return None };
        let Some(Column::Num { data, present, .. }) = store.column(key) else {
            return None;
        };
        specs.push((data, present, op, *t));
    }
    let Col::Nodes(ids) = batch.slot(slot0?) else {
        return None;
    };
    Some(
        ids.iter()
            .enumerate()
            .filter(|&(_, &id)| {
                let i = id as usize;
                specs
                    .iter()
                    .any(|&(data, present, op, t)| present[i] && num_pred(op, data[i], t))
            })
            .map(|(row, _)| row)
            .collect(),
    )
}

/// Keep rows satisfying a CONJUNCTION of `prop <op> num-literal` compares, all on
/// the same node slot and all reading `Num` columns — one raw-f64 pass over the id
/// list, each conjunct a `num_pred` (a NULL/NaN cell fails its conjunct → the row
/// drops, matching AND's 3VL). `None` unless every atom fits that shape (the
/// caller then tries the single-compare / general paths).
fn try_num_conjunction(pred: &Expr, store: &Store, batch: &Batch) -> Option<Vec<usize>> {
    if !matches!(pred, Expr::And(..)) {
        return None; // a single compare is handled by the caller's typed path
    }
    let mut atoms = Vec::new();
    flatten_and(pred, &mut atoms);
    let mut slot0: Option<usize> = None;
    let mut specs: Vec<(&[f64], &[bool], CompareOp, f64)> = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let Expr::Compare { op, left, right } = atom else {
            return None;
        };
        let (slot, key, op, lit) = match (left.as_ref(), right.as_ref()) {
            (Expr::Prop { slot, key }, Expr::Lit(v)) => (*slot, key, *op, v),
            (Expr::Lit(v), Expr::Prop { slot, key }) => (*slot, key, flip_op(*op), v),
            _ => return None,
        };
        match slot0 {
            Some(s) if s != slot => return None, // all atoms on the same slot
            _ => slot0 = Some(slot),
        }
        let Value::Num(t) = lit else { return None };
        let Some(Column::Num { data, present, .. }) = store.column(key) else {
            return None;
        };
        specs.push((data, present, op, *t));
    }
    let Col::Nodes(ids) = batch.slot(slot0?) else {
        return None;
    };
    // Same-column range (`lo <= x AND x < hi`) — the overwhelmingly common
    // conjunction. Normalize the two bounds to a concrete lower/upper with
    // inclusivity, then run ONE loop of LITERAL f64 comparisons: no per-element `match
    // op` and no runtime spec loop. NaN fails both compares (dropped), matching
    // `num_pred`'s 3VL; `present` gates nulls — byte-identical to the general path
    // below. At 1M this turns range filter+project from 0.68x to 1.10x.
    //
    // The 200k cache-resident `scan/range-and` PROJECTION sits at ~0.85x, and that is
    // projection-bound, not filter-bound: the FILTER, streamed, is 3.67x core (see
    // `try_stream_num_count` — the win was skipping the scan-id materialization, core's
    // trick), but this shape returns 20k names and the ~0.66ms of string projection
    // dominates the ~0.12ms filter, so both engines pay it and the ratio parks near a
    // tie. REJECTED, all measured NEUTRAL at 200k:
    //   - Streaming the projection too (collect survivors by borrowing the bucket, then
    //     project): neutral for the range (projection-bound) and it REGRESSED the
    //     single-compare `scan/gt` 1.05x -> 0.78x (a lone compare loses the vectorized
    //     `try_filter_keep` path for no materialization win).
    //   - Sequential column read when the id list is the contiguous full scan
    //     (`id == row`), to drop the `d0[ids[row]]` gather: no change.
    //   - Mask-then-compact: the default release target has no SIMD gather, so it would
    //     not vectorize either.
    if specs.len() == 2 {
        let (d0, p0, op0, t0) = specs[0];
        let (d1, _, op1, t1) = specs[1];
        if std::ptr::eq(d0.as_ptr(), d1.as_ptr()) {
            let bound = |op, t| match op {
                CompareOp::Ge => Some((true, t, true)), // (is_lower, value, inclusive)
                CompareOp::Gt => Some((true, t, false)),
                CompareOp::Le => Some((false, t, true)),
                CompareOp::Lt => Some((false, t, false)),
                _ => None, // Eq/Ne is not a range bound
            };
            if let (Some((lo_a, va, ia)), Some((lo_b, vb, ib))) = (bound(op0, t0), bound(op1, t1)) {
                // One bound must be lower and the other upper (else e.g. `x>=5 AND x>=10`,
                // not a range — fall through to the general path).
                let lohi = match (lo_a, lo_b) {
                    (true, false) => Some(((va, ia), (vb, ib))),
                    (false, true) => Some(((vb, ib), (va, ia))),
                    _ => None,
                };
                if let Some(((lo, lo_inc), (hi, hi_inc))) = lohi {
                    macro_rules! range_loop {
                        ($lo_cmp:tt, $hi_cmp:tt) => {{
                            let mut keep = Vec::new();
                            for (row, &id) in ids.iter().enumerate() {
                                let i = id as usize;
                                let x = d0[i];
                                if p0[i] && x $lo_cmp lo && x $hi_cmp hi {
                                    keep.push(row);
                                }
                            }
                            keep
                        }};
                    }
                    let keep = match (lo_inc, hi_inc) {
                        (true, true) => range_loop!(>=, <=),
                        (true, false) => range_loop!(>=, <),
                        (false, true) => range_loop!(>, <=),
                        (false, false) => range_loop!(>, <),
                    };
                    return Some(keep);
                }
            }
        }
    }
    Some(
        ids.iter()
            .enumerate()
            .filter(|&(_, &id)| {
                let i = id as usize;
                specs
                    .iter()
                    .all(|&(data, present, op, t)| present[i] && num_pred(op, data[i], t))
            })
            .map(|(row, _)| row)
            .collect(),
    )
}

/// Raw Str/Dict scan for `col STARTS/ENDS/CONTAINS lit`, keeping matches (`negate=false`)
/// or the complement (`negate=true`, i.e. `NOT (…)`). Semantics match `str_bool`: a present
/// string cell tests `f` (or `!f`); an absent/NULL cell is UNKNOWN under the inner test and
/// dropped EITHER way — the present-guard stays, so `NOT` does not resurrect absent rows,
/// exactly as `eval_mask`'s `Not(Some(false)) = Some(true)`, `Not(None) = None` does. A
/// non-string / absent-everywhere column yields no match (empty) in both directions. `None`
/// when the predicate is not this shape (caller falls to the general path).
fn try_keep_strsearch(
    pred: &Expr,
    store: &Store,
    batch: &Batch,
    negate: bool,
) -> Option<Vec<usize>> {
    let Expr::Call { name, args } = pred else {
        return None;
    };
    let (test, slot, key, sub) = match (name.as_str(), args.as_slice()) {
        (
            t @ ("starts_with" | "ends_with" | "contains"),
            [Expr::Prop { slot, key }, Expr::Lit(Value::Str(sub))],
        ) => (t, *slot, key, sub),
        _ => return None,
    };
    let Col::Nodes(ids) = batch.slot(slot) else {
        return None;
    };
    let f: fn(&str, &str) -> bool = match test {
        "starts_with" => |s, t| s.starts_with(t),
        "ends_with" => |s, t| s.ends_with(t),
        _ => |s, t| s.contains(t),
    };
    let sub = sub.as_ref();
    let mut keep = Vec::new();
    match store.column(key) {
        Some(Column::Str { data, present, .. }) => {
            for (row, &id) in ids.iter().enumerate() {
                let i = id as usize;
                if present[i] && (f(data[i].as_ref(), sub) != negate) {
                    keep.push(row);
                }
            }
            Some(keep)
        }
        Some(Column::Dict {
            dict,
            codes,
            present,
            ..
        }) => {
            for (row, &id) in ids.iter().enumerate() {
                let i = id as usize;
                if present[i] && (f(dict[codes[i] as usize].as_ref(), sub) != negate) {
                    keep.push(row);
                }
            }
            Some(keep)
        }
        // A boxed (de-opted) column may still hold STRING cells per row — a Str/Dict
        // column that a null or type-mixed write promoted to `Gen`. Test each string
        // cell; a non-string or absent cell is UNKNOWN → dropped (both directions),
        // matching `str_bool`. (Skipping this returned NO rows for a de-opted column.)
        // Column absent everywhere → every cell null → dropped, no error (both directions).
        None => Some(Vec::new()),
        // A Gen (mixed) or non-string typed column (Num/Bool/…) may hold a non-null
        // non-string value, on which STARTS/ENDS/CONTAINS now FAULTS (a type error, not a
        // silent no-match). Defer to the general evaluator so it raises the exception — an
        // all-string/null column yields the same rows there, only without this fast path.
        _ => None,
    }
}

fn try_filter_keep(pred: &Expr, store: &Store, batch: &Batch) -> Option<Vec<usize>> {
    // A predicate that is a pure function of ONE dict column: evaluate per distinct value
    // (≤ dict size) and keep by code, instead of a boxed eval per row.
    if let Some(keep) = try_filter_keep_dict(pred, store, batch) {
        return Some(keep);
    }
    // A numeric compare on an EDGE property (`e0.w <op> lit`): read the edge column by each
    // row's edge id in one raw-f64 pass. Without this an edge predicate is not fast-pathable,
    // so a conjunction like `c.name = 'x' AND e0.w <> 680` fell off the intersection path onto
    // the boxed general eval over the whole materialized 2-hop frontier. Absent → dropped
    // (NULL compare is UNKNOWN), matching the general filter.
    if let Expr::Compare { op, left, right } = pred {
        let bound = match (left.as_ref(), right.as_ref()) {
            (Expr::Prop { slot, key }, Expr::Lit(Value::Num(t))) => Some((*slot, key, *op, *t)),
            (Expr::Lit(Value::Num(t)), Expr::Prop { slot, key }) => {
                Some((*slot, key, flip_op(*op), *t))
            }
            _ => None,
        };
        if let Some((slot, key, op, t)) = bound {
            if let (Col::Edges(eids), Some((data, present))) =
                (batch.slot(slot), store.edge_num_column(key))
            {
                return Some(
                    eids.iter()
                        .enumerate()
                        .filter(|&(_, &eid)| {
                            let i = eid as usize;
                            present[i] && num_pred(op, data[i], t)
                        })
                        .map(|(row, _)| row)
                        .collect(),
                );
            }
        }
    }
    // `NOT p` pushes into the raw fast paths by inverting `p` (De Morgan + operator
    // flip), exact for the keep-TRUE filter. If the inverted form is not itself
    // fast-pathable, this returns None and the caller evaluates the original `NOT`
    // through the general (boxed) path.
    if let Expr::Not(inner) = pred {
        // `has(k, neq(v))` (and any negated `has(k, <cmp>)`) desugars to
        // `Not(And(PropertyExists{k}, Compare{k}))`. Per node the keep is `absent OR
        // !cmp` — a raw Num pass (an absent node IS kept) — instead of the general boxed
        // eval, which materialized ~all rows for a non-selective complement like neq.
        if let Expr::And(a, b) = inner.as_ref() {
            // Identify (PropertyExists{k}, Compare on k) in either order.
            let pair = match (a.as_ref(), b.as_ref()) {
                (pe @ Expr::PropertyExists { .. }, cmp @ Expr::Compare { .. })
                | (cmp @ Expr::Compare { .. }, pe @ Expr::PropertyExists { .. }) => Some((pe, cmp)),
                _ => None,
            };
            if let Some((
                Expr::PropertyExists { slot: s1, key: k1 },
                Expr::Compare { op, left, right },
            )) = pair
            {
                let bound = match (left.as_ref(), right.as_ref()) {
                    (Expr::Prop { slot, key }, Expr::Lit(Value::Num(t)))
                        if key == k1 && slot == s1 =>
                    {
                        Some((*op, *t))
                    }
                    (Expr::Lit(Value::Num(t)), Expr::Prop { slot, key })
                        if key == k1 && slot == s1 =>
                    {
                        Some((flip_op(*op), *t))
                    }
                    _ => None,
                };
                if let (Some((op, t)), Col::Nodes(ids), Some(Column::Num { data, present, .. })) =
                    (bound, batch.slot(*s1), store.column(k1))
                {
                    return Some(
                        ids.iter()
                            .enumerate()
                            .filter(|&(_, &id)| {
                                let i = id as usize;
                                !present[i] || !num_pred(op, data[i], t)
                            })
                            .map(|(row, _)| row)
                            .collect(),
                    );
                }
            }
        }
        // `NOT (col STARTS/ENDS/CONTAINS lit)` — the same raw Str/Dict scan as the
        // positive case, keeping the COMPLEMENT. `invert_pred` cannot turn a search Call
        // into a positive fast form, so without this the negation fell to the boxed eval
        // over the whole hop (the recurring `NOT ends_with` + projection loss).
        if let Some(keep) = try_keep_strsearch(inner, store, batch, true) {
            return Some(keep);
        }
        return invert_pred(inner).and_then(|pos| try_filter_keep(&pos, store, batch));
    }
    // A CONJUNCTION of typed-numeric `prop <op> literal` compares on one node slot
    // (e.g. `age >= 30 AND age < 40`) keeps rows satisfying ALL, in one raw-f64
    // pass — no per-cell boxing, and no falling to the general And evaluator.
    if let Some(keep) = try_num_conjunction(pred, store, batch) {
        return Some(keep);
    }
    // The OR mirror — `age < 5 OR age > 95`, and `age IN [1, 2, …]` (an OR-chain).
    if let Some(keep) = try_num_disjunction(pred, store, batch) {
        return Some(keep);
    }
    // A general MIXED conjunction: keep rows satisfying BOTH conjuncts by intersecting
    // each one's fast-path keep-set (both built in ascending row order, so a linear
    // merge). This is what generalizes `try_num_conjunction` (all-numeric) to the shape
    // a projection creates — `values(k)` / `valueMap` AND-s a `PropertyExists{k}` onto
    // the user's selective filter, and `And(PropertyExists, age = 90)` used to knock the
    // WHOLE filter off the raw path onto the boxed general eval over the full scan (the
    // selective-filter-then-projection cliff). Only fires when EVERY conjunct is itself
    // fast-pathable; otherwise `None` → the general path (no worse than before).
    if let Expr::And(a, b) = pred {
        // Evaluate the likely-more-SELECTIVE conjunct first, then test the other only on
        // its survivors — so a selective `age = 90` (100 of 100k) makes the second pass
        // 100 rows, not another 100k. A bare `PropertyExists` is a presence gate that
        // rarely reduces, so it goes LAST. Reduces the two-full-pass intersection to one
        // selective pass + a tiny follow-up.
        let a_gate = matches!(a.as_ref(), Expr::PropertyExists { .. });
        let b_gate = matches!(b.as_ref(), Expr::PropertyExists { .. });
        let (first, second) = if a_gate && !b_gate { (b, a) } else { (a, b) };
        let kf = try_filter_keep(first, store, batch)?;
        let sub = batch.gather(&kf);
        let ks = try_filter_keep(second, store, &sub)?;
        return Some(ks.iter().map(|&j| kf[j]).collect());
    }
    // `PropertyExists{k}` (the presence gate `values(k)` / element maps add): keep rows
    // whose column `k` is present — a raw `present[]` pass, no boxing. An
    // absent-everywhere column keeps nothing.
    if let Expr::PropertyExists { slot, key } = pred {
        let Col::Nodes(ids) = batch.slot(*slot) else {
            return None;
        };
        // Present = `present_at` (a typed value OR a stored present-null via the column's
        // nulls bit), so a stored null is counted.
        let Some(column) = store.column(key) else {
            return Some(Vec::new());
        };
        return Some(
            ids.iter()
                .enumerate()
                .filter(|&(_, &id)| column.present_at(id as usize))
                .map(|(row, _)| row)
                .collect(),
        );
    }
    // A string-search predicate `col STARTS WITH / ENDS WITH / CONTAINS lit` (which
    // desugars to a `starts_with`/`ends_with`/`contains` call) over a raw Str/Dict
    // column — scan `&str` directly, no per-cell `Value` boxing through `call_scalar`.
    if let Some(keep) = try_keep_strsearch(pred, store, batch, false) {
        return Some(keep);
    }
    let Expr::Compare { op, left, right } = pred else {
        return None;
    };
    let (slot, key, op, lit) = match (left.as_ref(), right.as_ref()) {
        (Expr::Prop { slot, key }, Expr::Lit(v)) => (*slot, key, *op, v),
        (Expr::Lit(v), Expr::Prop { slot, key }) => (*slot, key, flip_op(*op), v),
        _ => return None,
    };
    let Col::Nodes(ids) = batch.slot(slot) else {
        return None;
    };
    // A NULL literal makes every comparison UNKNOWN — no row is kept.
    if lit.is_null() {
        return Some(Vec::new());
    }
    let Some(column) = store.column(key) else {
        return Some(Vec::new()); // property absent everywhere → UNKNOWN → all dropped
    };
    let mut keep = Vec::new();
    // Typed fast path: a Num column vs a Num literal compares RAW f64 — no per-cell
    // `Value` boxing (the eval-vs-columnar cost). Semantics match the general
    // `compare`: ordering is 3VL (a NaN cell is unordered → dropped, via `<`/`>`
    // being false on NaN); equality via `==`/`!=`.
    if let (Column::Num { data, present, .. }, Value::Num(t)) = (column, lit) {
        let t = *t;
        for (row, &id) in ids.iter().enumerate() {
            let i = id as usize;
            if !present[i] {
                continue; // NULL → UNKNOWN → dropped
            }
            let x = data[i];
            let hit = match op {
                CompareOp::Eq => x == t,
                CompareOp::Ne => x != t,
                CompareOp::Lt => x < t,
                CompareOp::Le => x <= t,
                CompareOp::Gt => x > t,
                CompareOp::Ge => x >= t,
            };
            if hit {
                keep.push(row);
            }
        }
        return Some(keep);
    }
    // Typed fast path: a Str column vs a Str literal compares `&str` directly — no
    // per-cell `Value` boxing. `=`/`<>` are byte equality (== `value::equals`);
    // ordering is lexicographic (== `cmp_partial` for two strings). A NULL cell is
    // gated by `present`; a NULL literal was handled above.
    if let (Column::Str { data, present, .. }, Value::Str(t)) = (column, lit) {
        let t = t.as_ref();
        for (row, &id) in ids.iter().enumerate() {
            let i = id as usize;
            if !present[i] {
                continue; // NULL → UNKNOWN → dropped
            }
            let x = data[i].as_ref();
            let hit = match op {
                CompareOp::Eq => x == t,
                CompareOp::Ne => x != t,
                CompareOp::Lt => x < t,
                CompareOp::Le => x <= t,
                CompareOp::Gt => x > t,
                CompareOp::Ge => x >= t,
            };
            if hit {
                keep.push(row);
            }
        }
        return Some(keep);
    }
    // Typed fast path: a DICTIONARY-encoded string column vs a Str literal. `=`/`<>`
    // resolve the literal's code ONCE and compare `u32` codes per row (no per-cell string
    // read); ordering decodes `dict[code]` (a cheap indexed lookup, still no boxing). A
    // categorical column (`city`/`status`) is Dict, so this is what keeps a
    // `has('city','oslo').count()` off the boxed general path (which read a `Value::Str`
    // per node — the whole scan).
    if let (
        Column::Dict {
            dict,
            codes,
            present,
            ..
        },
        Value::Str(t),
    ) = (column, lit)
    {
        let t = t.as_ref();
        // For =/<> the answer is purely code equality against the literal's code (absent
        // from the dict ⇒ nothing equals it / everything present is unequal).
        if matches!(op, CompareOp::Eq | CompareOp::Ne) {
            let target = dict.iter().position(|d| d.as_ref() == t);
            for (row, &id) in ids.iter().enumerate() {
                let i = id as usize;
                if !present[i] {
                    continue;
                }
                let eq = target.is_some_and(|tc| codes[i] as usize == tc);
                if (op == CompareOp::Eq) == eq {
                    keep.push(row);
                }
            }
            return Some(keep);
        }
        for (row, &id) in ids.iter().enumerate() {
            let i = id as usize;
            if !present[i] {
                continue;
            }
            let x = dict[codes[i] as usize].as_ref();
            let hit = match op {
                CompareOp::Lt => x < t,
                CompareOp::Le => x <= t,
                CompareOp::Gt => x > t,
                CompareOp::Ge => x >= t,
                _ => unreachable!(), // Eq/Ne handled above
            };
            if hit {
                keep.push(row);
            }
        }
        return Some(keep);
    }
    // General path (Bool/Temporal/Gen columns): read the cell, then compare via the
    // value contract. Ordering uses `cmp_partial` (3VL — cross-type/NaN → drop,
    // matching `compare`), NOT the total order.
    for (row, &id) in ids.iter().enumerate() {
        let v = column.read(id as usize);
        if v.is_null() {
            continue;
        }
        let hit = match op {
            CompareOp::Eq => value::equals(&v, lit),
            CompareOp::Ne => !value::equals(&v, lit),
            _ => match value::cmp_partial(&v, lit) {
                Some(o) => match op {
                    CompareOp::Lt => o.is_lt(),
                    CompareOp::Le => o.is_le(),
                    CompareOp::Gt => o.is_gt(),
                    CompareOp::Ge => o.is_ge(),
                    _ => unreachable!("Eq/Ne handled above"),
                },
                None => continue, // incomparable → UNKNOWN → dropped
            },
        };
        if hit {
            keep.push(row);
        }
    }
    Some(keep)
}

/// Read `key` off an element frontier as a column, bulk-gathering the typed
/// storage column and staying unboxed when it and every read entry are
/// present-and-typed; fall to `Gen` (with nulls) otherwise.
fn read_property(store: &Store, col: &Col, key: &str) -> Col {
    // An edge slot reads an EDGE property (boxed map, keyed by eid). A `u32::MAX`
    // eid is the OPTIONAL null sentinel → NULL.
    if let Col::Edges(eids) = col {
        // Fastest path: the typed numeric overlay — read `data[eid]` as a raw f64 with
        // NO per-edge hash probe (the boxed `map.get` below). Only when every edge has
        // a present value (the null sentinel `u32::MAX` indexes past `present`, so it
        // fails the check and falls through to the null-carrying general column).
        if let Some((data, present)) = store.edge_num_column(key) {
            if eids
                .iter()
                .all(|&e| (e as usize) < present.len() && present[e as usize])
            {
                return Col::Num(eids.iter().map(|&e| data[e as usize]).collect());
            }
        }
        // Fast path: a fully-present NUMERIC edge property → a raw `Col::Num`, so the
        // downstream compare / aggregate hits the unboxed f64 path (the same win the
        // node columns already get). One outer hash lookup for `key`, then a probe
        // per edge; bail to the boxed `Gen` path the moment any edge is missing the
        // key, is the OPTIONAL null sentinel (`u32::MAX`, absent from the map), or is
        // non-numeric — those need the null-carrying general column.
        if let Some(map) = store.edge_prop_map(key) {
            let mut nums = Vec::with_capacity(eids.len());
            let ok = eids.iter().all(|&e| match map.get(&e) {
                Some(Value::Num(x)) => {
                    nums.push(*x);
                    true
                }
                _ => false,
            });
            if ok {
                return Col::Num(nums);
            }
        }
        return Col::Gen(
            eids.iter()
                .map(|&e| {
                    if e == u32::MAX {
                        Value::Null
                    } else {
                        store.edge_prop(e, key)
                    }
                })
                .collect(),
        );
    }
    // A node column carrying any OPTIONAL null sentinel reads per row (sentinel →
    // NULL, else the stored property); u32::MAX would index the property column out
    // of bounds on the fast path below.
    if let Col::Nodes(ids) = col {
        if ids.contains(&u32::MAX) {
            return Col::Gen(
                ids.iter()
                    .map(|&id| {
                        if id == u32::MAX {
                            Value::Null
                        } else {
                            store.prop(id, key)
                        }
                    })
                    .collect(),
            );
        }
    }
    let Col::Nodes(ids) = col else {
        // A non-element column (e.g. a projected Record): `x.key` reads the record
        // field; anything else has no property and reads NULL.
        return Col::Gen(
            (0..col.len())
                .map(|i| match col.value_at(i) {
                    Value::Record(fields) => value::record_field(&fields, key),
                    // A Map `.key` reads the entry under the string key `key`.
                    Value::Map(pairs) => pairs
                        .iter()
                        .find(|(k, _)| matches!(k, Value::Str(s) if s.as_ref() == key))
                        .map_or(Value::Null, |(_, v)| v.clone()),
                    _ => Value::Null,
                })
                .collect(),
        );
    };
    let Some(column) = store.column(key) else {
        return Col::Gen(vec![Value::Null; ids.len()]);
    };
    // Gather `data_at(i)` for every id in ONE pass, bailing to `None` the moment a value
    // is absent — so a fully-present column (the common case) does a single scattered
    // pass instead of the separate `all(present)` pre-check + gather (two passes over the
    // frontier). A null-bearing column bails and falls to the general per-row path.
    fn gather<T>(ids: &[u32], present: &[bool], data_at: impl Fn(usize) -> T) -> Option<Vec<T>> {
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            let i = id as usize;
            if !present[i] {
                return None;
            }
            out.push(data_at(i));
        }
        Some(out)
    }
    let general = || Col::Gen(ids.iter().map(|&i| store.prop(i, key)).collect());
    match column {
        Column::Num { data, present, .. } => {
            gather(ids, present, |i| data[i]).map_or_else(general, Col::Num)
        }
        Column::Str { data, present, .. } => {
            gather(ids, present, |i| data[i].clone()).map_or_else(general, Col::Str)
        }
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            gather(ids, present, |i| dict[codes[i] as usize].clone()).map_or_else(general, Col::Str)
        }
        Column::Bool { data, present, .. } => {
            gather(ids, present, |i| data[i]).map_or_else(general, Col::Bool)
        }
        _ => general(),
    }
}

fn broadcast(v: Value, n: usize) -> Col {
    match v {
        Value::Num(x) => Col::Num(vec![x; n]),
        Value::Bool(b) => Col::Bool(vec![b; n]),
        Value::Str(s) => Col::Str(vec![s; n]),
        // Null and List have no unboxed column form.
        other => Col::Gen(vec![other; n]),
    }
}

/// Compare two columns elementwise into a `Bool` column. `=`/`<>` use the value
/// contract's `equals`; ordering uses `cmp_total`. A NULL operand yields UNKNOWN,
/// carried as a `Gen` cell of `Null` so the three-valued logic upstream sees it.
fn compare(op: CompareOp, l: &Col, r: &Col) -> Col {
    let n = l.len().min(r.len());
    let mut out = Vec::with_capacity(n);
    let mut any_unknown = false;
    for i in 0..n {
        let a = l.value_at(i);
        let b = r.value_at(i);
        if a.is_null() || b.is_null() {
            any_unknown = true;
            out.push(None);
            continue;
        }
        // Equality uses the value contract's `equals` (cross-type = false, not unknown).
        // Ordering uses `cmp_partial`: a genuinely incomparable pair — DIFFERENT types — is
        // UNKNOWN (→ NULL). A NaN operand is IEEE, NOT 3VL: `<`/`>`/`<=`/`>=` are definitely
        // FALSE (matching JS and the pure-TS engine). Two Nums are incomparable ONLY via a
        // NaN, so `None` there → Some(false); `None` across types stays UNKNOWN.
        let order = |f: fn(std::cmp::Ordering) -> bool| -> Option<bool> {
            match value::cmp_partial(&a, &b) {
                Some(ord) => Some(f(ord)),
                None if matches!((&a, &b), (Value::Num(_), Value::Num(_))) => Some(false),
                None => None,
            }
        };
        let res = match op {
            CompareOp::Eq => Some(value::equals(&a, &b)),
            CompareOp::Ne => Some(!value::equals(&a, &b)),
            CompareOp::Lt => order(std::cmp::Ordering::is_lt),
            CompareOp::Le => order(std::cmp::Ordering::is_le),
            CompareOp::Gt => order(std::cmp::Ordering::is_gt),
            CompareOp::Ge => order(std::cmp::Ordering::is_ge),
        };
        if res.is_none() {
            any_unknown = true;
        }
        out.push(res);
    }
    if any_unknown {
        Col::Gen(
            out.into_iter()
                .map(|o| o.map_or(Value::Null, Value::Bool))
                .collect(),
        )
    } else {
        Col::Bool(out.into_iter().map(|o| o.expect("no unknowns")).collect())
    }
}

/// Read a column as three-valued booleans (None = UNKNOWN).
/// Coerce a non-boolean predicate column to Kleene truth for WHERE/FILTER, matching
/// core's `as_truth`: a NUMBER is true when non-zero and non-NaN; a STRING when
/// non-empty; NULL is unknown (`None`); any other non-null value (temporal, list,
/// record, element) is true. (A bare `WHERE <number>` / `WHERE <string>` is thus a
/// truthiness test, not a no-match — the engine used to drop every non-bool row.)
/// The message for a non-boolean used where a truth value is required (WHERE / FILTER /
/// CASE WHEN / AND / OR / NOT / XOR). A number or string is NOT coerced to a truth value
/// (SQL / ISO-GQL); the only path to a boolean is an explicit CAST AS BOOLEAN / to_boolean.
const TRUTH_TYPE_ERR: &str =
    "E_INVALID_VALUE: a boolean is required — a non-boolean value is not coerced to a truth \
     value (use CAST(x AS BOOLEAN) or to_boolean(x))";

/// Three-valued truth of each cell in a boolean context. A present non-null NON-boolean
/// is a data exception (strict typing — no truthiness coercion); NULL is UNKNOWN (`None`).
fn as_truth(col: &Col) -> Result<Vec<Option<bool>>, String> {
    match col {
        Col::Bool(bs) => Ok(bs.iter().map(|&b| Some(b)).collect()),
        other => (0..other.len())
            .map(|i| match other.value_at(i) {
                Value::Null => Ok(None),
                Value::Bool(b) => Ok(Some(b)),
                _ => Err(TRUTH_TYPE_ERR.to_string()),
            })
            .collect(),
    }
}

fn map_bool(col: &Col, f: impl Fn(Option<bool>) -> Option<bool>) -> Result<Col, String> {
    Ok(truth_to_col(as_truth(col)?.into_iter().map(f).collect()))
}

/// A vectorized three-valued predicate mask over `batch` (`Some(true)`/`Some(false)`/
/// `None` = UNKNOWN). The boolean connectives combine their operands' masks with the SAME
/// Kleene tables `eval` uses, and a numeric leaf `prop <cmp> lit` reads the typed column
/// directly instead of boxing two `Value`s per row through `compare`. Any other
/// sub-expression falls back to the boxed `eval` for that node only, so the result is
/// exactly `as_truth(eval(expr))` — this removes allocation/boxing on the boolean spine
/// and the numeric leaves, nothing more. Used by the filter keep-set (a complex predicate
/// that `try_filter_keep` declines) and the reverse-seed residual.
fn eval_mask(expr: &Expr, store: &Store, batch: &Batch) -> Result<Vec<Option<bool>>, String> {
    Ok(match expr {
        Expr::Not(x) => eval_mask(x, store, batch)?
            .into_iter()
            .map(|o| o.map(|b| !b))
            .collect(),
        Expr::And(l, r) => {
            let (a, b) = (eval_mask(l, store, batch)?, eval_mask(r, store, batch)?);
            let n = a.len().min(b.len());
            (0..n)
                .map(|i| match (a[i], b[i]) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                })
                .collect()
        }
        Expr::Or(l, r) => {
            let (a, b) = (eval_mask(l, store, batch)?, eval_mask(r, store, batch)?);
            let n = a.len().min(b.len());
            (0..n)
                .map(|i| match (a[i], b[i]) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                })
                .collect()
        }
        Expr::Xor(l, r) => {
            let (a, b) = (eval_mask(l, store, batch)?, eval_mask(r, store, batch)?);
            let n = a.len().min(b.len());
            (0..n)
                .map(|i| match (a[i], b[i]) {
                    (Some(x), Some(y)) => Some(x != y),
                    _ => None,
                })
                .collect()
        }
        Expr::Compare { op, left, right } => {
            if let Some(m) = typed_num_mask(*op, left, right, store, batch) {
                m
            } else if let Some(m) = typed_str_mask(*op, left, right, store, batch) {
                m
            } else {
                as_truth(&eval(expr, store, batch)?)?
            }
        }
        Expr::Call { name, args } => match typed_strsearch_mask(name, args, store, batch) {
            Some(m) => m,
            None => as_truth(&eval(expr, store, batch)?)?,
        },
        _ => as_truth(&eval(expr, store, batch)?)?,
    })
}

/// A numeric leaf `prop <cmp> lit` (or its mirror) over a node/edge frontier → typed mask,
/// reading the Num column raw. `None` when the leaf is not a Num-column-vs-num-literal. A
/// present Num cell is always finite (NaN/Inf are stored as NULL), so `num_pred` matches
/// the boxed `compare`'s three-valued result exactly; an absent cell is UNKNOWN.
fn typed_num_mask(
    op: CompareOp,
    left: &Expr,
    right: &Expr,
    store: &Store,
    batch: &Batch,
) -> Option<Vec<Option<bool>>> {
    let (slot, key, op, t) = match (left, right) {
        (Expr::Prop { slot, key }, Expr::Lit(Value::Num(t))) => (*slot, key, op, *t),
        (Expr::Lit(Value::Num(t)), Expr::Prop { slot, key }) => (*slot, key, flip_op(op), *t),
        _ => return None,
    };
    match batch.slot(slot) {
        Col::Nodes(ids) => {
            let Some(Column::Num { data, present, .. }) = store.column(key) else {
                return None;
            };
            Some(
                ids.iter()
                    .map(|&id| {
                        let i = id as usize;
                        present[i].then(|| num_pred(op, data[i], t))
                    })
                    .collect(),
            )
        }
        Col::Edges(eids) => {
            let (data, present) = store.edge_num_column(key)?;
            Some(
                eids.iter()
                    .map(|&eid| {
                        let i = eid as usize;
                        present[i].then(|| num_pred(op, data[i], t))
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

/// A string leaf `prop <cmp> lit` (or its mirror) over a node frontier → typed mask,
/// reading the `Str`/`Dict` column raw (no per-row `Value`). Equality/inequality on a
/// `Dict` column compares interned codes (code equality ⟺ string equality); ordering and
/// `Str` columns compare the strings directly. `None` when not a string-column-vs-string-
/// literal; an absent cell is UNKNOWN. Cross-type (`Str` prop vs non-`Str` lit) returns
/// `None`, so the boxed `compare` keeps its cross-type semantics.
fn typed_str_mask(
    op: CompareOp,
    left: &Expr,
    right: &Expr,
    store: &Store,
    batch: &Batch,
) -> Option<Vec<Option<bool>>> {
    let (slot, key, op, lit) = match (left, right) {
        (Expr::Prop { slot, key }, Expr::Lit(Value::Str(s))) => (*slot, key, op, s),
        (Expr::Lit(Value::Str(s)), Expr::Prop { slot, key }) => (*slot, key, flip_op(op), s),
        _ => return None,
    };
    let Col::Nodes(ids) = batch.slot(slot) else {
        return None;
    };
    let lit: &str = lit.as_ref();
    match store.column(key)? {
        Column::Str { data, present, .. } => Some(
            ids.iter()
                .map(|&id| {
                    let i = id as usize;
                    present[i].then(|| str_pred(op, data[i].as_ref(), lit))
                })
                .collect(),
        ),
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } if matches!(op, CompareOp::Eq | CompareOp::Ne) => {
            // Code compare: a present cell matches iff its code equals the literal's; a
            // literal absent from the dict matches nothing (eq→false, ne→true).
            let lit_code = dict.iter().position(|s| s.as_ref() == lit);
            let want_eq = matches!(op, CompareOp::Eq);
            Some(
                ids.iter()
                    .map(|&id| {
                        let i = id as usize;
                        present[i].then(|| match lit_code {
                            Some(lc) => (codes[i] as usize == lc) == want_eq,
                            None => !want_eq,
                        })
                    })
                    .collect(),
            )
        }
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => Some(
            ids.iter()
                .map(|&id| {
                    let i = id as usize;
                    present[i].then(|| str_pred(op, dict[codes[i] as usize].as_ref(), lit))
                })
                .collect(),
        ),
        _ => None,
    }
}

/// A string-search leaf `prop STARTS WITH / ENDS WITH / CONTAINS lit` (a `Call` over a
/// node `Str`/`Dict` column) → typed mask, scanning `&str` directly instead of boxing each
/// cell through `str_bool`. `None` when it is not that shape; a present string cell tests,
/// an absent cell — or a non-string / missing column — is UNKNOWN (matching `str_bool`'s
/// NULL), so the mask equals the boxed result row-for-row.
fn typed_strsearch_mask(
    name: &str,
    args: &[Expr],
    store: &Store,
    batch: &Batch,
) -> Option<Vec<Option<bool>>> {
    let (Expr::Prop { slot, key }, Expr::Lit(Value::Str(sub))) = (args.first()?, args.get(1)?)
    else {
        return None;
    };
    let f: fn(&str, &str) -> bool = match name {
        "starts_with" => |s, t| s.starts_with(t),
        "ends_with" => |s, t| s.ends_with(t),
        "contains" => |s, t| s.contains(t),
        _ => return None,
    };
    let Col::Nodes(ids) = batch.slot(*slot) else {
        return None;
    };
    let sub = sub.as_ref();
    // Only a pure Str/Dict column (every present cell is a string) can be vectorized here.
    // An ABSENT column is NULL for every row → null-propagation, no error (matches the
    // general evaluator). A Gen (mixed) or a non-string typed column may hold a non-null
    // non-string cell, on which the predicate now FAULTS — return None so `eval_mask`
    // falls to the general `eval`, which raises the type error per row (identical to the
    // function-form `starts_with(...)`).
    match store.column(key) {
        None => Some(vec![None; ids.len()]),
        Some(Column::Str { data, present, .. }) => Some(
            ids.iter()
                .map(|&id| {
                    let i = id as usize;
                    present[i].then(|| f(data[i].as_ref(), sub))
                })
                .collect(),
        ),
        Some(Column::Dict {
            dict,
            codes,
            present,
            ..
        }) => Some(
            ids.iter()
                .map(|&id| {
                    let i = id as usize;
                    present[i].then(|| f(dict[codes[i] as usize].as_ref(), sub))
                })
                .collect(),
        ),
        _ => None,
    }
}

fn zip_bool(
    store: &Store,
    batch: &Batch,
    l: &Expr,
    r: &Expr,
    f: impl Fn(Option<bool>, Option<bool>) -> Option<bool>,
) -> Result<Col, String> {
    let lc = as_truth(&eval(l, store, batch)?)?;
    let rc = as_truth(&eval(r, store, batch)?)?;
    let n = lc.len().min(rc.len());
    Ok(truth_to_col((0..n).map(|i| f(lc[i], rc[i])).collect()))
}

fn truth_to_col(out: Vec<Option<bool>>) -> Col {
    if out.iter().all(Option::is_some) {
        Col::Bool(out.into_iter().map(|o| o.expect("all some")).collect())
    } else {
        Col::Gen(
            out.into_iter()
                .map(|o| o.map_or(Value::Null, Value::Bool))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Plan;
    use crate::store::Builder;
    use std::sync::Arc;

    /// The iterative `varlen_walk` must emit the exact same paths, in the exact same
    /// order, as the recursive `varlen_dfs` it replaced — over every mode/direction/bound
    /// combination on a spread of random graphs. Byte-identity is the hard invariant; this
    /// is the direct A/B guard (the corpus + differential fuzzer cover the predicate hooks).
    #[test]
    fn iterative_varlen_matches_recursive() {
        struct RecordEmit {
            paths: Vec<(Vec<u32>, Vec<u32>)>,
        }
        impl VarlenEmit for RecordEmit {
            fn emit(&mut self, _row: usize, node_stack: &[u32], edge_stack: &[u32]) {
                self.paths.push((node_stack.to_vec(), edge_stack.to_vec()));
            }
            fn should_stop(&self) -> bool {
                false
            }
        }
        #[allow(clippy::too_many_arguments)]
        fn collect(
            store: &Store,
            n_nodes: u32,
            mode: PathMode,
            dir: Dir,
            min: u32,
            max: u32,
            k: u32,
            double_loops: bool,
            iterative: bool,
        ) -> Vec<(Vec<u32>, Vec<u32>)> {
            let node_unique = matches!(mode, PathMode::Simple | PathMode::Acyclic);
            let mut sink = RecordEmit { paths: Vec::new() };
            let mut used: Vec<u32> = Vec::new();
            for v in 0..n_nodes {
                if node_unique {
                    used.push(v);
                }
                let mut ns = vec![v];
                let mut es: Vec<u32> = Vec::new();
                if iterative {
                    varlen_walk(
                        store,
                        v,
                        min,
                        max,
                        dir,
                        &[],
                        mode,
                        v,
                        &mut used,
                        v as usize,
                        &mut ns,
                        &mut es,
                        None,
                        k,
                        None,
                        None,
                        double_loops,
                        &mut sink,
                    );
                } else {
                    varlen_dfs(
                        store,
                        v,
                        0,
                        min,
                        max,
                        dir,
                        &[],
                        mode,
                        v,
                        &mut used,
                        v as usize,
                        &mut ns,
                        &mut es,
                        None,
                        k,
                        None,
                        None,
                        double_loops,
                        &mut sink,
                    );
                }
                if node_unique {
                    used.pop();
                }
                assert!(used.is_empty(), "used stack left dirty after a source");
            }
            sink.paths
        }

        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _trial in 0..60 {
            let n_nodes = 4 + (rng() % 7) as u32;
            let mut nd = String::new();
            for i in 0..n_nodes {
                nd.push_str(&format!(
                    "{{\"id\":\"n{i}\",\"labels\":[\"P\"],\"props\":{{}}}}\n"
                ));
            }
            let ecount = (rng() % (u64::from(n_nodes) * 3 + 1)) as u32;
            for e in 0..ecount {
                let f = (rng() % u64::from(n_nodes)) as u32;
                let t = (rng() % u64::from(n_nodes)) as u32;
                nd.push_str(&format!(
                    "{{\"id\":\"e{e}\",\"from\":\"n{f}\",\"to\":\"n{t}\",\"labels\":[\"R\"],\"props\":{{}}}}\n"
                ));
            }
            let store = crate::ndjson::from_ndjson(&nd).unwrap();
            for mode in [
                PathMode::Walk,
                PathMode::Trail,
                PathMode::Simple,
                PathMode::Acyclic,
            ] {
                for dir in [Dir::Out, Dir::In, Dir::Both] {
                    for (min, max, k) in [
                        (0u32, 3u32, 1u32),
                        (1, 4, 1),
                        (2, 2, 1),
                        (1, 6, 2),
                        (0, 4, 1),
                    ] {
                        for double_loops in [false, true] {
                            let rec = collect(
                                &store,
                                n_nodes,
                                mode,
                                dir,
                                min,
                                max,
                                k,
                                double_loops,
                                false,
                            );
                            let itr = collect(
                                &store,
                                n_nodes,
                                mode,
                                dir,
                                min,
                                max,
                                k,
                                double_loops,
                                true,
                            );
                            assert_eq!(
                                rec, itr,
                                "mode={mode:?} dir={dir:?} {min}..={max} k={k} dl={double_loops} ecount={ecount}"
                            );
                            // The count/agg fast-path twins (k=1, no preds) must equal the
                            // materialized rows: count == #rows, agg-sum == sum of endpoints.
                            if k == 1 {
                                let node_unique =
                                    matches!(mode, PathMode::Simple | PathMode::Acyclic);
                                let mut total = 0u64;
                                let mut used: Vec<u32> = Vec::new();
                                for src in 0..n_nodes {
                                    if node_unique {
                                        used.push(src);
                                    }
                                    varlen_count_dfs(
                                        &store,
                                        src,
                                        0,
                                        min,
                                        max,
                                        dir,
                                        &[],
                                        mode,
                                        src,
                                        &mut used,
                                        &mut total,
                                        double_loops,
                                    );
                                    if node_unique {
                                        used.pop();
                                    }
                                }
                                assert_eq!(
                                    total as usize,
                                    itr.len(),
                                    "count twin vs materialized rows: mode={mode:?} dir={dir:?} {min}..={max} dl={double_loops}"
                                );
                                // The agg fast-path is only taken without a both()-doubled
                                // self-loop, so compare it against the dl=false rows only.
                                if !double_loops {
                                    let mut sum = 0u64;
                                    let mut used2: Vec<u32> = Vec::new();
                                    for src in 0..n_nodes {
                                        if node_unique {
                                            used2.push(src);
                                        }
                                        varlen_agg_dfs(
                                            &store,
                                            src,
                                            0,
                                            min,
                                            max,
                                            dir,
                                            &[],
                                            mode,
                                            src,
                                            &mut used2,
                                            &mut |v| sum += u64::from(v),
                                        );
                                        if node_unique {
                                            used2.pop();
                                        }
                                    }
                                    let want: u64 = itr
                                        .iter()
                                        .map(|(ns, _)| u64::from(*ns.last().unwrap()))
                                        .sum();
                                    assert_eq!(
                                        sum, want,
                                        "agg-sum twin vs materialized endpoints: mode={mode:?} dir={dir:?} {min}..={max}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The iterative walk uses O(1) call stack regardless of closure depth: a 40k-deep
    /// traversal — which the old recursive DFS would have driven ~40k frames deep, blowing
    /// any normal stack — completes on a deliberately TINY (512 KiB) thread. This is why a
    /// deep closure can no longer overflow (and why its peak memory is now bounded heap,
    /// not committed stack). Drives `run_varlen` directly, off any big stack.
    #[test]
    fn deep_varlen_walk_runs_on_a_tiny_stack() {
        // A 40k-long chain n0->n1->...; recursing it would need ~20 MB of stack.
        let mut nd = String::new();
        for i in 0..40_000u32 {
            nd.push_str(&format!(
                "{{\"id\":\"n{i}\",\"labels\":[\"P\"],\"props\":{{}}}}\n"
            ));
        }
        for i in 0..39_999u32 {
            nd.push_str(&format!(
                "{{\"id\":\"e{i}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"props\":{{}}}}\n",
                i + 1
            ));
        }
        let store = crate::ndjson::from_ndjson(&nd).unwrap();
        // Drive the walk directly (bypassing the pull machinery) so the test isolates
        // varlen_walk's OWN stack use. An unbounded out-walk from n0 emits one path per
        // reachable prefix (39 999 of them) and recurses 40k deep in the old DFS.
        struct CountEmit(usize);
        impl VarlenEmit for CountEmit {
            fn emit(&mut self, _row: usize, _node_stack: &[u32], _edge_stack: &[u32]) {
                self.0 += 1;
            }
            fn should_stop(&self) -> bool {
                false
            }
        }
        let handle = std::thread::Builder::new()
            .stack_size(512 * 1024) // far below the ~20 MB a recursive DFS would need
            .spawn(move || {
                let mut sink = CountEmit(0);
                run_varlen(
                    &[0], // source = n0
                    &store,
                    &[],      // any edge label
                    1,        // min
                    u32::MAX, // max (unbounded)
                    Dir::Out,
                    PathMode::Walk,
                    None,
                    1, // k
                    None,
                    None,
                    false,
                    &mut sink,
                );
                // The count fast-path twin must ALSO be O(1) stack (it shares varlen_scan_walk).
                let mut total = 0u64;
                let mut used: Vec<u32> = Vec::new();
                varlen_count_dfs(
                    &store,
                    0,
                    0,
                    1,
                    u32::MAX,
                    Dir::Out,
                    &[],
                    PathMode::Walk,
                    0,
                    &mut used,
                    &mut total,
                    false,
                );
                (sink.0, total)
            })
            .unwrap();
        assert_eq!(
            handle.join().expect("must not overflow the tiny stack"),
            (39_999, 39_999)
        );
    }

    fn n(x: f64) -> Value {
        Value::Num(x)
    }
    fn s(x: &str) -> Value {
        Value::Str(Arc::from(x))
    }
    fn prop(slot: usize, key: &str) -> Expr {
        Expr::Prop {
            slot,
            key: key.to_string(),
        }
    }
    fn lit(v: Value) -> Expr {
        Expr::Lit(v)
    }
    fn cmp(op: CompareOp, l: Expr, r: Expr) -> Expr {
        Expr::Compare {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }
    fn scan(label: &str) -> Plan {
        Plan::Scan {
            label: Some(label.to_string()),
        }
    }
    fn names_of(out: &Rows, col: usize) -> Vec<String> {
        out.rows
            .iter()
            .map(|r| match &r[col] {
                Value::Str(x) => x.to_string(),
                other => format!("{other:?}"),
            })
            .collect()
    }

    fn social() -> Store {
        let mut b = Builder::default();
        let a = b.node(&["Person"], &[("name", s("alice")), ("age", n(30.0))]);
        let bob = b.node(&["Person"], &[("name", s("bob")), ("age", n(25.0))]);
        let c = b.node(&["Person"], &[("name", s("carol")), ("age", n(40.0))]);
        let proj = b.node(&["Project"], &[("name", s("graphdb"))]);
        b.edge(a, bob, "KNOWS");
        b.edge(a, c, "KNOWS");
        b.edge(bob, c, "KNOWS");
        b.edge(a, proj, "WORKS_ON");
        b.build()
    }

    /// The opt-in edge-type index is a pure optimization: a type-filtered hop
    /// returns the SAME rows with it on as with it off (for_each_nbr routes to the
    /// bucket, but the answer is identical).
    #[test]
    fn edge_type_index_gives_identical_query_results() {
        let mut store = social();
        let plan = crate::gql::parse("MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name AS b").unwrap();
        let mut before = names_of(&run(&plan, &store), 0);
        before.sort();
        store.create_edge_type_index();
        let mut after = names_of(&run(&plan, &store), 0);
        after.sort();
        assert_eq!(before, after);
        // alice KNOWS bob & carol; bob KNOWS carol → bob, carol, carol.
        assert_eq!(after, vec!["bob", "carol", "carol"]);
        // A count through the fused fast path also matches.
        let cplan =
            crate::gql::parse("MATCH (a:Person)-[:KNOWS]->() RETURN count(*) AS c").unwrap();
        assert!(matches!(run(&cplan, &store).rows[0][0], Value::Num(x) if x == 3.0));
    }

    /// A store whose only node named "target" (n0) is reachable by several 2- and
    /// 3-hop R-paths, plus decoy paths that never reach it. Used to prove the
    /// multi-hop reverse-seed returns the SAME multiset as the forward walk.
    fn reverse_seed_store() -> Store {
        let mut b = Builder::default();
        let t = b.node(&["N"], &[("name", s("target"))]);
        let m1 = b.node(&["N"], &[("name", s("m1"))]);
        let m2 = b.node(&["N"], &[("name", s("m2"))]);
        let s3 = b.node(&["N"], &[("name", s("s3"))]);
        let s4 = b.node(&["N"], &[("name", s("s4"))]);
        let r8 = b.node(&["N"], &[("name", s("r8"))]);
        let r9 = b.node(&["N"], &[("name", s("r9"))]);
        // decoy chain that never reaches the target
        let d0 = b.node(&["N"], &[("name", s("other"))]);
        let d1 = b.node(&["N"], &[("name", s("d1"))]);
        let d2 = b.node(&["N"], &[("name", s("d2"))]);
        b.edge(m1, t, "R");
        b.edge(m2, t, "R");
        b.edge(s3, m1, "R");
        b.edge(s3, m2, "R"); // s3 reaches target two ways (diamond)
        b.edge(s4, m1, "R");
        b.edge(r8, s3, "R");
        b.edge(r9, s4, "R");
        b.edge(d1, d0, "R"); // decoys
        b.edge(d2, d1, "R");
        b.build()
    }

    /// The multi-hop reverse-seed (an indexed selective endpoint over an Expand chain)
    /// returns exactly the rows the forward walk does — same multiset, index on or off,
    /// at two and three hops. Index off ⇒ forward; index on ⇒ seed-and-reverse.
    #[test]
    fn reverse_seed_multihop_matches_forward() {
        let mut st = reverse_seed_store();
        let two = "MATCH (a)-[:R]->(b)-[:R]->(c) WHERE c.name = 'target' RETURN a.name AS a";
        let three =
            "MATCH (a)-[:R]->(b)-[:R]->(c)-[:R]->(d) WHERE d.name = 'target' RETURN a.name AS a";
        let sorted = |st: &Store, q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
            v.sort();
            v
        };

        // Forward (no index): 2-hop reaches target via m1 (from s3,s4) and m2 (from s3).
        let fwd2 = sorted(&st, two);
        assert_eq!(fwd2, vec!["s3", "s3", "s4"]);
        let fwd3 = sorted(&st, three); // r8→s3→{m1,m2}→t, r9→s4→m1→t
        assert_eq!(fwd3, vec!["r8", "r8", "r9"]);

        // Index on: the reverse-seed fires and must return the identical multiset.
        st.create_index("name");
        assert_eq!(sorted(&st, two), fwd2);
        assert_eq!(sorted(&st, three), fwd3);

        // count(*) rides the same seed and matches the row count.
        let cnt = |q: &str| match run(&crate::gql::parse(q).unwrap(), &st).rows[0][0] {
            Value::Num(x) => x,
            _ => panic!("count not numeric"),
        };
        assert_eq!(
            cnt("MATCH (a)-[:R]->(b)-[:R]->(c) WHERE c.name = 'target' RETURN count(*) AS c"),
            3.0
        );
        assert_eq!(
            cnt("MATCH (a)-[:R]->(b)-[:R]->(c)-[:R]->(d) WHERE d.name = 'target' RETURN count(*) AS c"),
            3.0
        );
    }

    /// A keyless `LIMIT` over a reverse-seeded chain (the OrderPage fast path) returns
    /// the same rows the forward walk + LIMIT does — capped below the result size, and
    /// unchanged above it — index on or off. Guards against the OrderPage stream path
    /// silently bypassing the seed.
    #[test]
    fn reverse_seed_under_limit_matches_forward() {
        let mut st = reverse_seed_store();
        let q = |lim: usize| {
            format!("MATCH (a)-[:R]->(b)-[:R]->(c) WHERE c.name = 'target' RETURN a.name AS a LIMIT {lim}")
        };
        let rows =
            |st: &Store, lim: usize| run(&crate::gql::parse(&q(lim)).unwrap(), st).rows.len();
        // Forward (no index): 3 matching rows, so LIMIT 2 caps to 2, LIMIT 10 keeps 3.
        assert_eq!(rows(&st, 2), 2);
        assert_eq!(rows(&st, 10), 3);
        st.create_index("name"); // reverse-seed now fires under the LIMIT
        assert_eq!(rows(&st, 2), 2);
        assert_eq!(rows(&st, 10), 3);
        // Above the result size the full multiset must match the forward walk exactly.
        let mut got = names_of(&run(&crate::gql::parse(&q(10)).unwrap(), &st), 0);
        got.sort();
        assert_eq!(got, vec!["s3", "s3", "s4"]);
    }

    /// The reverse VAR-LENGTH seed returns the forward walk's exact multiset — including
    /// duplicate-path multiplicity (s3 reaches the target two ways at length 2). Index
    /// off ⇒ forward var-length; index on ⇒ seed the endpoint and walk the quantifier
    /// backward. Covers a low and a high hop window.
    #[test]
    fn reverse_varlen_seed_matches_forward() {
        let mut st = reverse_seed_store();
        let cases: [(&str, Vec<&str>); 2] = [
            (
                "MATCH (a)-[:R]->{1,2}(b) WHERE b.name = 'target' RETURN a.name AS a",
                vec!["m1", "m2", "s3", "s3", "s4"],
            ),
            (
                "MATCH (a)-[:R]->{2,3}(b) WHERE b.name = 'target' RETURN a.name AS a",
                vec!["r8", "r8", "r9", "s3", "s3", "s4"],
            ),
        ];
        let sorted = |st: &Store, q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
            v.sort();
            v
        };
        // Forward (no index) matches the hand-computed multiset.
        for (q, want) in &cases {
            assert_eq!(&sorted(&st, q), want, "forward {q}");
        }
        // Index on → the reverse var-length seed fires and returns the identical multiset.
        st.create_index("name");
        for (q, want) in &cases {
            assert_eq!(&sorted(&st, q), want, "reverse {q}");
        }
    }

    /// A store for the COMPOUND reverse var-length: a fixed `-[:F]->` hop feeds an R
    /// var-length to a unique "target". `x` reaches the target two ways at length 2, so
    /// `a1` (which F-reaches `x`) must appear with that multiplicity.
    fn compound_varlen_store() -> Store {
        let mut b = Builder::default();
        let t = b.node(&["N"], &[("name", s("target"))]);
        let m1 = b.node(&["N"], &[("name", s("m1"))]);
        let m2 = b.node(&["N"], &[("name", s("m2"))]);
        let x = b.node(&["N"], &[("name", s("x"))]);
        let a1 = b.node(&["N"], &[("name", s("a1"))]);
        let a2 = b.node(&["N"], &[("name", s("a2"))]);
        b.edge(m1, t, "R");
        b.edge(m2, t, "R");
        b.edge(x, m1, "R");
        b.edge(x, m2, "R"); // x reaches target two ways at length 2
        b.edge(a1, m1, "F");
        b.edge(a2, m2, "F");
        b.edge(a1, x, "F");
        b.build()
    }

    /// The reverse var-length seed behind a leading fixed hop returns the forward walk's
    /// exact multiset. Reversal walks the var-length back to the fixed hop's target, then
    /// the fixed hop back to the labeled source — `a1` appears three times at {1,2}
    /// (via m1 once, via x twice) and twice at {2,3} (both length-2 paths through x).
    #[test]
    fn reverse_varlen_compound_matches_forward() {
        let mut st = compound_varlen_store();
        let cases: [(&str, Vec<&str>); 2] = [
            (
                "MATCH (a)-[:F]->(v)-[:R]->{1,2}(c) WHERE c.name = 'target' RETURN a.name AS a",
                vec!["a1", "a1", "a1", "a2"],
            ),
            (
                "MATCH (a)-[:F]->(v)-[:R]->{2,3}(c) WHERE c.name = 'target' RETURN a.name AS a",
                vec!["a1", "a1"],
            ),
        ];
        let sorted = |st: &Store, q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
            v.sort();
            v
        };
        for (q, want) in &cases {
            assert_eq!(&sorted(&st, q), want, "forward {q}");
        }
        st.create_index("name"); // compound reverse var-length fires
        for (q, want) in &cases {
            assert_eq!(&sorted(&st, q), want, "reverse {q}");
        }
    }

    /// `DISTINCT <low-card dict prop> LIMIT n` with `n` above the distinct count returns
    /// the same rows as the uncapped DISTINCT (the LIMIT is a no-op) — the fast path must
    /// not drop or reorder values.
    #[test]
    fn distinct_dict_limit_noop_matches_uncapped() {
        let mut b = Builder::default();
        for i in 0..40u32 {
            b.node(&["N"], &[("c", s(["x", "y", "z"][(i % 3) as usize]))]);
        }
        let st = b.build();
        let rows = |q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
            v.sort();
            v
        };
        let full = rows("MATCH (n) RETURN DISTINCT n.c AS x");
        assert_eq!(full, vec!["x", "y", "z"]);
        // LIMIT 10 > 3 distinct → no-op; the vectorized fast path yields the same set.
        assert_eq!(rows("MATCH (n) RETURN DISTINCT n.c AS x LIMIT 10"), full);
        // A binding LIMIT still caps (3 distinct, LIMIT 2 → 2 rows).
        assert_eq!(
            run(
                &crate::gql::parse("MATCH (n) RETURN DISTINCT n.c AS x LIMIT 2").unwrap(),
                &st
            )
            .rows
            .len(),
            2
        );
    }

    /// The vectorized filter mask (`eval_mask`) keeps three-valued (Kleene) logic exact for
    /// a complex predicate over a NULL-bearing column: an UNKNOWN row is dropped, `OR` with a
    /// TRUE is TRUE even when the other side is UNKNOWN, and `NOT UNKNOWN` stays UNKNOWN.
    #[test]
    fn eval_mask_three_valued_semantics() {
        let mut b = Builder::default();
        b.node(&["N"], &[("age", n(60.0)), ("city", s("oslo"))]); // n0
        b.node(&["N"], &[("age", n(10.0)), ("city", s("bergen"))]); // n1
        b.node(&["N"], &[("city", s("oslo"))]); // n2: age absent (null)
        b.node(&["N"], &[("city", s("bergen"))]); // n3: age absent
        let st = b.build();
        let cities = |q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
            v.sort();
            v
        };
        // n0 age>50=T; n1 both F; n2 null OR city=T → T; n3 null OR F → null (dropped).
        assert_eq!(
            cities("MATCH (n) WHERE (n.age > 50 OR n.city = 'oslo') RETURN n.city AS c"),
            vec!["oslo", "oslo"]
        );
        // NOT of the above: n0 F, n1 T, n2 F, n3 NOT null = null (dropped).
        assert_eq!(
            cities("MATCH (n) WHERE NOT (n.age > 50 OR n.city = 'oslo') RETURN n.city AS c"),
            vec!["bergen"]
        );
    }

    /// A string-search leaf (`STARTS WITH`/`ENDS WITH`/`CONTAINS`) inside a complex
    /// predicate keeps three-valued semantics through `eval_mask`: a null string cell is
    /// UNKNOWN (dropped, or `NOT UNKNOWN` = UNKNOWN), matching the boxed `str_bool`.
    #[test]
    fn eval_mask_string_search_three_valued() {
        let mut b = Builder::default();
        b.node(&["N"], &[("name", s("apple")), ("city", s("oslo"))]); // n0
        b.node(&["N"], &[("name", s("banana")), ("city", s("bergen"))]); // n1
        b.node(&["N"], &[("city", s("bergen"))]); // n2: name null
        let st = b.build();
        let cities = |q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
            v.sort();
            v
        };
        // n0 name STARTS 'a' = T; n1 F or 'bergen' ENDS 'o' = F; n2 null OR F = null (drop).
        assert_eq!(
            cities("MATCH (n) WHERE (n.name STARTS WITH 'a' OR n.city ENDS WITH 'o') RETURN n.city AS c"),
            vec!["oslo"]
        );
        // NOT (name STARTS 'a'): n0 F, n1 T, n2 NOT null = null (drop).
        assert_eq!(
            cities("MATCH (n) WHERE NOT (n.name STARTS WITH 'a') RETURN n.city AS c"),
            vec!["bergen"]
        );
    }

    /// A chain that BINDS an edge (`-[e:R]->`) with an edge-property residual reverse-seeds
    /// correctly: the reverse-walk must capture each hop's edge and land it in the right
    /// column, so both the edge residual (`e.w < 100`) and a `RETURN e.w` see the true edge.
    #[test]
    fn reverse_bound_edge_matches_forward() {
        let mut bd = Builder::default();
        let t = bd.node(&["N"], &[("name", s("target"))]);
        let m1 = bd.node(&["N"], &[("name", s("m1"))]);
        let m2 = bd.node(&["N"], &[("name", s("m2"))]);
        let a1 = bd.node(&["N"], &[("name", s("a1"))]);
        let a2 = bd.node(&["N"], &[("name", s("a2"))]);
        bd.edge(m1, t, "R"); // eid 0
        bd.edge(m2, t, "R"); // eid 1
        bd.edge(a1, m1, "F"); // eid 2
        bd.edge(a2, m2, "F"); // eid 3
        let mut st = bd.build();
        st.set_edge_prop(0, "w", n(5.0));
        st.set_edge_prop(1, "w", n(500.0));
        let names = |st: &Store, q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
            v.sort();
            v
        };
        // a1→m1-(w5)→target passes `e.w < 100`; a2→m2-(w500) fails.
        let q = "MATCH (a)-[:F]->(m)-[e:R]->(c) WHERE c.name = 'target' AND e.w < 100 RETURN a.name AS a";
        assert_eq!(names(&st, q), vec!["a1"]);
        st.create_index("name");
        assert_eq!(names(&st, q), vec!["a1"]); // reverse-seed with the edge residual applied

        // The bound-edge column must carry the actual edges (RETURN reads slot 2 = edge).
        let qe = "MATCH (a)-[:F]->(m)-[e:R]->(c) WHERE c.name = 'target' RETURN e.w AS w";
        let mut ws: Vec<f64> = run(&crate::gql::parse(qe).unwrap(), &st)
            .rows
            .iter()
            .map(|r| match r[0] {
                Value::Num(x) => x,
                _ => panic!("w not numeric"),
            })
            .collect();
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(ws, vec![5.0, 500.0]);
    }

    /// The generalized seeds — range (`>`), positive `IN`, and `OR` of seedables — each
    /// return the forward walk's exact multiset (the `OR` case has s1 twice: it reaches an
    /// age-matched endpoint and a city-matched one). Forward with no index; seeded with a
    /// hash index (equality/IN) and a range index (range) present.
    #[test]
    fn reverse_range_in_or_seeds_match_forward() {
        let mut b = Builder::default();
        let e1 = b.node(
            &["N"],
            &[("name", s("e1")), ("age", n(95.0)), ("city", s("oslo"))],
        );
        let e2 = b.node(
            &["N"],
            &[("name", s("e2")), ("age", n(99.0)), ("city", s("bergen"))],
        );
        let e3 = b.node(
            &["N"],
            &[("name", s("e3")), ("age", n(50.0)), ("city", s("oslo"))],
        );
        let s1 = b.node(&["N"], &[("name", s("s1"))]);
        let s2 = b.node(&["N"], &[("name", s("s2"))]);
        let s3 = b.node(&["N"], &[("name", s("s3"))]);
        b.edge(s1, e1, "R");
        b.edge(s2, e2, "R");
        b.edge(s1, e3, "R");
        b.edge(s3, e1, "R");
        let mut st = b.build();
        let cases: [(&str, Vec<&str>); 3] = [
            (
                "MATCH (a)-[:R]->(b) WHERE b.age > 90 RETURN a.name AS a",
                vec!["s1", "s2", "s3"],
            ),
            (
                "MATCH (a)-[:R]->(b) WHERE b.age IN [95, 99] RETURN a.name AS a",
                vec!["s1", "s2", "s3"],
            ),
            (
                "MATCH (a)-[:R]->(b) WHERE (b.age > 90 OR b.city = 'oslo') RETURN a.name AS a",
                vec!["s1", "s1", "s2", "s3"],
            ),
        ];
        let sorted = |st: &Store, q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
            v.sort();
            v
        };
        for (q, want) in &cases {
            assert_eq!(&sorted(&st, q), want, "forward {q}");
        }
        st.create_index("age"); // IN over the hash index
        st.create_index("city"); // OR's equality disjunct
        st.create_range_index("age"); // range / OR's range disjunct
        for (q, want) in &cases {
            assert_eq!(&sorted(&st, q), want, "seeded {q}");
        }
    }

    /// Two range bounds on the SAME key seed their exact intersection via the two-sided
    /// range seek. A narrow interval keeps only the in-range endpoints, a contradictory
    /// pair (lo > hi) seeds the empty set, and a same-direction pair falls through to the
    /// generic per-conjunct seed — all matching the forward walk.
    #[test]
    fn reverse_seed_interval_intersection_matches_forward() {
        let mut b = Builder::default();
        let e1 = b.node(&["N"], &[("name", s("e1")), ("age", n(95.0))]);
        let e2 = b.node(&["N"], &[("name", s("e2")), ("age", n(99.0))]);
        let e3 = b.node(&["N"], &[("name", s("e3")), ("age", n(50.0))]);
        let s1 = b.node(&["N"], &[("name", s("s1"))]);
        let s2 = b.node(&["N"], &[("name", s("s2"))]);
        let s3 = b.node(&["N"], &[("name", s("s3"))]);
        b.edge(s1, e1, "R");
        b.edge(s2, e2, "R");
        b.edge(s3, e1, "R");
        b.edge(s2, e3, "R"); // e3 (age 50) is reachable but filtered out by every case
        let mut st = b.build();
        let cases: [(&str, Vec<&str>); 3] = [
            // narrow interval [>90, <98] → only e1 (95); e2=99 and e3=50 excluded
            (
                "MATCH (a)-[:R]->(b) WHERE (b.age > 90 AND b.age < 98) RETURN a.name AS a",
                vec!["s1", "s3"],
            ),
            // contradictory (>98 AND <90) → empty
            (
                "MATCH (a)-[:R]->(b) WHERE (b.age > 98 AND b.age < 90) RETURN a.name AS a",
                vec![],
            ),
            // same direction (>40 AND >60) → e1,e2 (e3=50 fails >60); generic per-conjunct seed
            (
                "MATCH (a)-[:R]->(b) WHERE (b.age > 40 AND b.age > 60) RETURN a.name AS a",
                vec!["s1", "s2", "s3"],
            ),
        ];
        let sorted = |st: &Store, q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
            v.sort();
            v
        };
        for (q, want) in &cases {
            assert_eq!(&sorted(&st, q), want, "forward {q}");
        }
        st.create_range_index("age");
        for (q, want) in &cases {
            assert_eq!(&sorted(&st, q), want, "seeded {q}");
        }
    }

    /// `NOT (x <op> v)` normalizes to `x <neg op> v` — the negated spelling returns exactly
    /// the positive one's rows, including the 3VL NULL case (an absent operand is UNKNOWN and
    /// dropped either way, never resurrected) and stacked `NOT NOT NOT` collapsing.
    #[test]
    fn negated_comparison_matches_positive_spelling() {
        let mut b = Builder::default();
        let n1 = b.node(&["N"], &[("name", s("keep")), ("age", n(30.0))]);
        let n2 = b.node(&["N"], &[("name", s("skip")), ("age", n(40.0))]);
        let n3 = b.node(&["N"], &[("name", s("noage"))]); // age ABSENT
        let src = b.node(&["N"], &[("name", s("src"))]);
        b.edge(src, n1, "R");
        b.edge(src, n2, "R");
        b.edge(src, n3, "R");
        let st = b.build();
        let sorted = |q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
            v.sort();
            v
        };
        // NOT (age <> 30) == age = 30 → keep n1; absent n3 is UNKNOWN, dropped both ways.
        assert_eq!(
            sorted("MATCH (a)-[:R]->(x) WHERE NOT x.age <> 30 RETURN x.name AS n"),
            sorted("MATCH (a)-[:R]->(x) WHERE x.age = 30 RETURN x.name AS n"),
        );
        assert_eq!(
            sorted("MATCH (a)-[:R]->(x) WHERE NOT x.age <> 30 RETURN x.name AS n"),
            vec!["keep"]
        );
        // NOT (age >= 40) == age < 40 → keep n1 (30); n2 (40) excluded; absent dropped.
        assert_eq!(
            sorted("MATCH (a)-[:R]->(x) WHERE NOT x.age >= 40 RETURN x.name AS n"),
            sorted("MATCH (a)-[:R]->(x) WHERE x.age < 40 RETURN x.name AS n"),
        );
        // Stacked negation collapses: NOT NOT NOT (age >= 40) == age < 40.
        assert_eq!(
            sorted("MATCH (a)-[:R]->(x) WHERE NOT NOT NOT x.age >= 40 RETURN x.name AS n"),
            vec!["keep"]
        );
    }

    /// `DISTINCT x, x` (identical projection items) routes to the single-column path and
    /// replicates the result column — same distinct set, and the second column is the exact
    /// replica of the first (not a separately-keyed composite).
    #[test]
    fn distinct_identical_columns_replicate() {
        let mut b = Builder::default();
        let m1 = b.node(&["N"], &[("name", s("m1"))]);
        let m2 = b.node(&["N"], &[("name", s("m2"))]);
        let t = b.node(&["N"], &[("name", s("t"))]);
        let src = b.node(&["N"], &[("name", s("src"))]);
        b.edge(src, m1, "R");
        b.edge(src, m2, "R");
        b.edge(src, t, "R");
        b.edge(m1, m2, "R"); // m1 also reaches m2 — forces the endpoint dedup
        let st = b.build();
        let batch = run(
            &crate::gql::parse("MATCH (a)-[:R]->(x) RETURN DISTINCT x.name AS p, x.name AS q")
                .unwrap(),
            &st,
        );
        let mut c0 = names_of(&batch, 0);
        let c1 = names_of(&batch, 1);
        assert_eq!(
            c0.clone()
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            c0.len(),
            "col0 is distinct"
        );
        c0.sort();
        let mut c1s = c1.clone();
        c1s.sort();
        assert_eq!(c0, vec!["m1", "m2", "t"]);
        assert_eq!(
            c1,
            names_of(&batch, 0),
            "second column replicates the first row-for-row"
        );
        assert_eq!(c0, c1s);
    }

    /// `X AND (Y OR Z)` where `X ∧ Z` is numerically contradictory drops the Z branch, and a
    /// NON-contradictory Z is preserved — both must return the SAME rows as the un-simplified
    /// predicate (the simplification is logically exact, not just a fast path).
    #[test]
    fn contradictory_or_branch_pruned_matches_semantics() {
        let mut b = Builder::default();
        let hi = b.node(&["N"], &[("name", s("hit")), ("age", n(50.0))]);
        let lo = b.node(&["N"], &[("name", s("hit")), ("age", n(10.0))]);
        let ms = b.node(&["N"], &[("name", s("miss")), ("age", n(50.0))]);
        let src = b.node(&["N"], &[("name", s("src"))]);
        b.edge(src, hi, "R");
        b.edge(src, lo, "R");
        b.edge(src, ms, "R");
        let st = b.build();
        let sorted = |q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
            v.sort();
            v
        };
        // age>=40 AND age<20 is contradictory → the OR collapses to name='hit'. Only hit(50)
        // satisfies age>=40 AND name='hit'; miss(50) fails the name, lo(10) fails the age.
        assert_eq!(
            sorted("MATCH (a)-[:R]->(b) WHERE (b.age >= 40 AND (b.name = 'hit' OR b.age < 20)) RETURN b.name AS n"),
            vec!["hit"]
        );
        // age>=45 is NOT contradictory with age>=40 → NO pruning: miss(50) satisfies the OR via
        // age>=45, so it MUST survive (guards against over-pruning).
        assert_eq!(
            sorted("MATCH (a)-[:R]->(b) WHERE (b.age >= 40 AND (b.name = 'hit' OR b.age >= 45)) RETURN b.name AS n"),
            vec!["hit", "miss"]
        );
    }

    /// The fused single-hop `count(*)` over a numeric-filtered typed hop counts exactly the
    /// (source, neighbour) PATHS whose neighbour passes the filter — the same value the
    /// materialize path yields — for BOTH a universal source label (flat-edge sweep) and a
    /// labelled SUBSET (per-source slice), never counting an edge from an out-of-label source.
    #[test]
    fn fused_hop_count_matches_materialize_value() {
        let count_of = |q: &str, st: &Store| -> f64 {
            let out = run(&crate::gql::parse(q).unwrap(), st);
            match &out.rows.iter().next().expect("one count row")[0] {
                Value::Num(n) => *n,
                other => panic!("not a number: {other:?}"),
            }
        };
        // Non-universal: p* are Person, o0 is Other. Only Person sources' F edges count.
        let mut b = Builder::default();
        let p0 = b.node(&["Person"], &[("age", n(50.0))]);
        let p1 = b.node(&["Person"], &[("age", n(80.0))]);
        let p2 = b.node(&["Person"], &[("age", n(90.0))]);
        let o0 = b.node(&["Other"], &[("age", n(30.0))]);
        b.edge(p0, p1, "F"); // 80 >= 60 ✓
        b.edge(p0, o0, "F"); // 30 >= 60 ✗
        b.edge(p1, p2, "F"); // 90 >= 60 ✓
        b.edge(o0, p1, "F"); // source o0 is NOT a Person → excluded
        let st = b.build();
        let q = "MATCH (a:Person)-[:F]->(b) WHERE b.age >= 60 RETURN count(*) AS c";
        assert_eq!(count_of(q, &st), 2.0, "labelled-subset path");

        // Universal: every node is a Person → the flat-edge sweep fires. Same graph, all Person.
        let mut b2 = Builder::default();
        let q0 = b2.node(&["Person"], &[("age", n(50.0))]);
        let q1 = b2.node(&["Person"], &[("age", n(80.0))]);
        let q2 = b2.node(&["Person"], &[("age", n(90.0))]);
        let q3 = b2.node(&["Person"], &[("age", n(30.0))]);
        b2.edge(q0, q1, "F");
        b2.edge(q0, q3, "F");
        b2.edge(q1, q2, "F");
        b2.edge(q3, q1, "F"); // now q3 IS a Person → this edge counts (target q1 age80 ✓)
        let st2 = b2.build();
        // targets passing age>=60: q0→q1(80✓), q0→q3(30✗), q1→q2(90✓), q3→q1(80✓) = 3.
        assert_eq!(count_of(q, &st2), 3.0, "universal flat-sweep path");
    }

    /// The 1-hop `count(*)` degree-sum reads raw adjacency lengths when the hop's type
    /// set covers EVERY edge type (the `matching_degree` fast path), and still filters
    /// by type when it does not — the counts must agree with the per-edge walk in both
    /// cases. Regression: this shape (`(a)-[:R]->(b) RETURN count(*)`) walked every edge
    /// with a per-edge type check, ~1.8x slower than core; the raw-length path fixes it
    /// WITHOUT changing the value.
    #[test]
    fn one_hop_count_uses_raw_degree_only_when_type_set_is_universal() {
        let count_of = |q: &str, st: &Store| -> f64 {
            match &run(&crate::gql::parse(q).unwrap(), st)
                .rows
                .iter()
                .next()
                .expect("one count row")[0]
            {
                Value::Num(n) => *n,
                other => panic!("not a number: {other:?}"),
            }
        };
        // Single edge type `R`: `[:R]` covers all types → raw-degree fast path.
        // Also a MULTI-type graph so a `[:R]` hop is a PARTIAL want (must still filter),
        // plus a directed self-loop (kept once by Out) and an unlabeled `-->` (any type).
        let mut b = Builder::default();
        let a = b.node(&["N"], &[]);
        let c = b.node(&["N"], &[]);
        let d = b.node(&["N"], &[]);
        b.edge(a, c, "R");
        b.edge(a, d, "R");
        b.edge(a, a, "R"); // directed self-loop: an out-edge counted once
        b.edge(c, d, "S"); // a SECOND edge type
        let st = b.build();
        // Out over R from every N: a has 3 R-out (c, d, self), c has 0 R-out → 3.
        assert_eq!(
            count_of("MATCH (x:N)-[:R]->(y) RETURN count(*) AS c", &st),
            3.0
        );
        // In over R: c←1 (a), d←1 (a), a←1 (self) → 3.
        assert_eq!(
            count_of("MATCH (x:N)<-[:R]-(y) RETURN count(*) AS c", &st),
            3.0
        );
        // Partial want `[:S]` in a multi-type graph: only the one S edge (c→d) → 1.
        assert_eq!(
            count_of("MATCH (x:N)-[:S]->(y) RETURN count(*) AS c", &st),
            1.0
        );
        // Anonymous edge `-[]->` = any type (empty want) → all 4 edges.
        assert_eq!(
            count_of("MATCH (x:N)-[]->(y) RETURN count(*) AS c", &st),
            4.0
        );
    }

    /// A two-column DISTINCT with a high-card Str column dedups on the (Str, other) tuple key
    /// exactly as the byte-key would: same distinct tuples, first-seen order, and a present-null
    /// component collapses with an absent one.
    #[test]
    fn str_composite_distinct_dedups_correctly() {
        let mut b = Builder::default();
        let src = b.node(&["N"], &[("name", s("src"))]);
        // (alice,30) twice via two neighbours, (alice,40) once, (bob,30) once → 3 distinct tuples.
        let n1 = b.node(&["N"], &[("name", s("alice")), ("age", n(30.0))]);
        let n2 = b.node(&["N"], &[("name", s("alice")), ("age", n(30.0))]);
        let n3 = b.node(&["N"], &[("name", s("alice")), ("age", n(40.0))]);
        let n4 = b.node(&["N"], &[("name", s("bob")), ("age", n(30.0))]);
        b.edge(src, n1, "R");
        b.edge(src, n2, "R");
        b.edge(src, n3, "R");
        b.edge(src, n4, "R");
        let st = b.build();
        let out = run(
            &crate::gql::parse("MATCH (a)-[:R]->(x) RETURN DISTINCT x.name AS n, x.age AS g")
                .unwrap(),
            &st,
        );
        let mut tuples: Vec<(String, String)> = out
            .rows
            .iter()
            .map(|r| (format!("{:?}", r[0]), format!("{:?}", r[1])))
            .collect();
        assert_eq!(tuples.len(), 3, "three distinct (name, age) tuples");
        tuples.sort();
        assert_eq!(
            tuples,
            vec![
                ("Str(\"alice\")".into(), "Num(30.0)".into()),
                ("Str(\"alice\")".into(), "Num(40.0)".into()),
                ("Str(\"bob\")".into(), "Num(30.0)".into()),
            ]
        );
    }

    /// The fused numeric-filtered projection returns the SAME rows (as a multiset) as the
    /// general materialize+filter+gather+project path — same survivors, same projected values.
    #[test]
    fn fused_hop_projection_matches_general() {
        let mut b = Builder::default();
        let src = b.node(&["Person"], &[("name", s("src"))]);
        for (sc, nm) in [
            (10.0, "a"),
            (60.0, "b"),
            (30.0, "c"),
            (90.0, "d"),
            (20.0, "e"),
        ] {
            let t = b.node(&["Person"], &[("score", n(sc)), ("name", s(nm))]);
            b.edge(src, t, "F");
        }
        let st = b.build();
        // score < 50 keeps a(10,→'a'), c(30,→'c'), e(20,→'e'); b,d dropped.
        let mut got = names_of(
            &run(
                &crate::gql::parse(
                    "MATCH (a:Person)-[:F]->(b) WHERE b.score < 50 RETURN b.name AS n",
                )
                .unwrap(),
                &st,
            ),
            0,
        );
        got.sort();
        assert_eq!(got, vec!["a", "c", "e"]);
        // A projected expression (not a bare prop) over the survivor frontier still works.
        let mut up = names_of(
            &run(
                &crate::gql::parse(
                    "MATCH (a:Person)-[:F]->(b) WHERE b.score < 50 RETURN upper(b.name) AS n",
                )
                .unwrap(),
                &st,
            ),
            0,
        );
        up.sort();
        assert_eq!(up, vec!["A", "C", "E"]);
    }

    /// The fused mask-aggregate (count/sum/min/max over a complex-predicate typed hop) returns
    /// the SAME scalar as the general materialize+filter+aggregate path — checked against the
    /// un-fused plan on the same graph, so any row-order or skip divergence would show.
    #[test]
    fn fused_mask_agg_matches_general_aggregate() {
        let mut b = Builder::default();
        let src = b.node(&["Person"], &[("name", s("src"))]);
        // targets with (score, age); the OR (score >= 50 OR age < 5) keeps some, drops others.
        for (sc, ag) in [
            (10.0, 3.0),
            (60.0, 90.0),
            (55.0, 2.0),
            (20.0, 40.0),
            (99.0, 10.0),
        ] {
            let t = b.node(&["Person"], &[("score", n(sc)), ("age", n(ag))]);
            b.edge(src, t, "F");
        }
        let st = b.build();
        let scalar = |q: &str| -> f64 {
            match &run(&crate::gql::parse(q).unwrap(), &st)
                .rows
                .iter()
                .next()
                .expect("one row")[0]
            {
                Value::Num(n) => *n,
                other => panic!("not a number: {other:?}"),
            }
        };
        // Kept scores: 60(≥50), 55(≥50 & age<5), 99(≥50), 10(age<5) → {60,55,99,10}. 20 dropped.
        let base = "MATCH (a:Person)-[:F]->(b) WHERE (b.score >= 50 OR b.age < 5)";
        assert_eq!(scalar(&format!("{base} RETURN count(*) AS c")), 4.0);
        assert_eq!(scalar(&format!("{base} RETURN sum(b.score) AS v")), 224.0);
        assert_eq!(scalar(&format!("{base} RETURN max(b.score) AS v")), 99.0);
        assert_eq!(scalar(&format!("{base} RETURN min(b.score) AS v")), 10.0);
    }

    /// A single-type hop over the per-type CSR returns the type's neighbours in the SAME
    /// order (and multiplicity) as the flat scan filtering on the edge type — the byte-identity
    /// the partition must preserve. Interleaves F and R out-edges from one source.
    #[test]
    fn per_type_hop_preserves_flat_scan_order() {
        let mut b = Builder::default();
        let src = b.node(&["N"], &[("name", s("src"))]);
        let f1 = b.node(&["N"], &[("name", s("f1"))]);
        let r1 = b.node(&["N"], &[("name", s("r1"))]);
        let f2 = b.node(&["N"], &[("name", s("f2"))]);
        let r2 = b.node(&["N"], &[("name", s("r2"))]);
        let f3 = b.node(&["N"], &[("name", s("f3"))]);
        // Interleave the two types in insertion order: F R F R F.
        b.edge(src, f1, "F");
        b.edge(src, r1, "R");
        b.edge(src, f2, "F");
        b.edge(src, r2, "R");
        b.edge(src, f3, "F");
        let st = b.build();
        // The F hop must yield f1, f2, f3 in insertion order (ORDER matters — no sort).
        let names = names_of(
            &run(
                &crate::gql::parse("MATCH (a)-[:F]->(b) RETURN b.name AS n").unwrap(),
                &st,
            ),
            0,
        );
        assert_eq!(names, vec!["f1", "f2", "f3"]);
        // And the R hop yields r1, r2 in insertion order.
        let rnames = names_of(
            &run(
                &crate::gql::parse("MATCH (a)-[:R]->(b) RETURN b.name AS n").unwrap(),
                &st,
            ),
            0,
        );
        assert_eq!(rnames, vec!["r1", "r2"]);
    }

    /// `NOT (col STARTS/ENDS/CONTAINS lit)` over a hop keeps the complement via the raw
    /// scan, and — critically — an ABSENT cell stays dropped (UNKNOWN under the inner
    /// search, so NOT-UNKNOWN is UNKNOWN), matching the general eval_mask path exactly.
    #[test]
    fn not_strsearch_keep_matches_general() {
        let mut b = Builder::default();
        let e1 = b.node(&["N"], &[("name", s("alpha")), ("city", s("oslo"))]);
        let e2 = b.node(&["N"], &[("name", s("beta")), ("city", s("bergen"))]);
        let e3 = b.node(&["N"], &[("name", s("gamma"))]); // city ABSENT
        let src = b.node(&["N"], &[("name", s("src"))]);
        b.edge(src, e1, "R");
        b.edge(src, e2, "R");
        b.edge(src, e3, "R");
        let st = b.build();
        let sorted = |q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
            v.sort();
            v
        };
        // ENDS WITH 'ta': only beta; NOT keeps alpha, gamma.
        assert_eq!(
            sorted("MATCH (a)-[:R]->(b) WHERE NOT b.name ENDS WITH 'ta' RETURN b.name AS n"),
            vec!["alpha", "gamma"]
        );
        // STARTS WITH 'a': only alpha; NOT keeps beta, gamma.
        assert_eq!(
            sorted("MATCH (a)-[:R]->(b) WHERE NOT b.name STARTS WITH 'a' RETURN b.name AS n"),
            vec!["beta", "gamma"]
        );
        // CONTAINS 'mm': only gamma; NOT keeps alpha, beta.
        assert_eq!(
            sorted("MATCH (a)-[:R]->(b) WHERE NOT b.name CONTAINS 'mm' RETURN b.name AS n"),
            vec!["alpha", "beta"]
        );
        // NOT city CONTAINS 'o': oslo fails, bergen keeps (beta), gamma's city ABSENT is
        // UNKNOWN → dropped (not resurrected by NOT).
        assert_eq!(
            sorted("MATCH (a)-[:R]->(b) WHERE NOT b.city CONTAINS 'o' RETURN b.name AS n"),
            vec!["beta"]
        );
    }

    /// A conjunction seeded on its equality conjunct (`c.name = 'hit' AND c.age > 50`)
    /// applies the remaining conjuncts as a residual filter over the seeded rows, so it
    /// returns exactly the forward walk's rows — the seed bucket holds two 'hit' nodes
    /// and only the one passing `age > 50` (and the paths reaching it) survive.
    #[test]
    fn reverse_seed_conjunction_residual_matches_forward() {
        let mut b = Builder::default();
        let t1 = b.node(&["N"], &[("name", s("hit")), ("age", n(99.0))]);
        let t2 = b.node(&["N"], &[("name", s("hit")), ("age", n(10.0))]); // same name, fails age>50
        let s1 = b.node(&["N"], &[("name", s("s1"))]);
        let s2 = b.node(&["N"], &[("name", s("s2"))]);
        b.edge(s1, t1, "R");
        b.edge(s2, t2, "R");
        b.edge(s1, t2, "R"); // s1 also reaches the filtered-out target
        let mut st = b.build();
        let q = "MATCH (a)-[:R]->(c) WHERE c.name = 'hit' AND c.age > 50 RETURN a.name AS a";
        let sorted = |st: &Store| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
            v.sort();
            v
        };
        // Forward (no index): only s1→t1 survives (t1.age = 99).
        assert_eq!(sorted(&st), vec!["s1"]);
        // Index on: seed name='hit' (bucket {t1,t2}), residual age>50 keeps only t1's path.
        st.create_index("name");
        assert_eq!(sorted(&st), vec!["s1"]);
    }

    /// The reverse-seed only fires when the target bucket is smaller than the source
    /// scan. A non-selective endpoint (every node named "x") must keep the forward
    /// walk — and either way the rows are identical.
    #[test]
    fn reverse_seed_declines_non_selective_endpoint() {
        let mut b = Builder::default();
        let ids: Vec<u32> = (0..6)
            .map(|_| b.node(&["N"], &[("name", s("x"))]))
            .collect();
        for w in ids.windows(2) {
            b.edge(w[0], w[1], "R");
        }
        let mut st = b.build();
        let q = "MATCH (a)-[:R]->(b)-[:R]->(c) WHERE c.name = 'x' RETURN a.name AS a";
        let fwd = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
        st.create_index("name"); // bucket = all 6 nodes >= source, so no flip
        let idx = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
        let (mut fwd, mut idx) = (fwd, idx);
        fwd.sort();
        idx.sort();
        assert_eq!(fwd, idx);
    }

    /// `r.vf <= X AND r.vt >= Y` fuses to an `IntervalExpand` whose scan fallback now
    /// compares TEMPORAL bounds (not just numeric) via the value contract — so a
    /// "contains the window" query over date edges returns the covering edges instead
    /// of nothing (the numeric-only guard used to skip every temporal edge).
    #[test]
    fn interval_expand_scan_handles_temporal_bounds() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"id\":\"covers\",\"vf\":{\"@date\":\"2024-01-01\"},\"vt\":{\"@date\":\"2024-12-01\"}}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"id\":\"exact\",\"vf\":{\"@date\":\"2024-04-01\"},\"vt\":{\"@date\":\"2024-08-01\"}}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"id\":\"disjoint\",\"vf\":{\"@date\":\"2024-01-01\"},\"vt\":{\"@date\":\"2024-03-01\"}}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let q = "MATCH ()-[r:R]->() WHERE r.vf <= DATE '2024-04-01' AND r.vt >= DATE '2024-08-01' RETURN r.id AS id ORDER BY id";
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        // The optimizer must have fused it into the interval hop.
        assert!(has_interval_expand(&plan));
        assert_eq!(names_of(&run(&plan, &store), 0), vec!["covers", "exact"]);
    }

    /// Does the plan tree contain an `IntervalExpand` (the fused interval hop)?
    fn has_interval_expand(p: &Plan) -> bool {
        match p {
            Plan::IntervalExpand { .. } => true,
            Plan::Sample { input, .. }
            | Plan::Enumerate { input, .. }
            | Plan::EdgeVertex { input, .. }
            | Plan::Expand { input, .. }
            | Plan::VarLength { input, .. }
            | Plan::ShortestPath { input, .. }
            | Plan::Filter { input, .. }
            | Plan::Aggregate { input, .. }
            | Plan::OrderPage { input, .. }
            | Plan::Project { input, .. }
            | Plan::Distinct { input }
            | Plan::SortLocal { input, .. }
            | Plan::Update { input, .. }
            | Plan::UpdateReturn { input, .. } => has_interval_expand(input),
            Plan::Join { left, right, .. } => {
                has_interval_expand(left) || has_interval_expand(right)
            }
            _ => false,
        }
    }

    fn interval_store() -> Store {
        // Emp 0 with 5 HELD edges to role 1, intervals [d, d+2] for d in 0..5.
        let mut b = Builder::default();
        b.node(&["Emp"], &[]);
        b.node(&["Role"], &[]);
        let mut st = b.build();
        for d in 0..5u32 {
            let e = st.add_edge(0, 1, "HELD");
            st.set_edge_prop(e, "vf", n(f64::from(d)));
            st.set_edge_prop(e, "vt", n(f64::from(d) + 2.0));
        }
        st
    }

    /// The optimizer fuses `r.vf <= X AND r.vt >= Y` over a bound-edge hop into an
    /// `IntervalExpand`, which returns the SAME rows via the scan fallback (no
    /// index) and via the index seek — and both equal the hand-computed answer.
    #[test]
    fn interval_expand_fuses_and_matches_scan_and_seek() {
        use crate::opt::optimize;
        let mut st = interval_store();
        // As of t=3: [0,2] no, [1,3] yes, [2,4] yes, [3,5] yes, [4,6] no → 3.
        let q = "MATCH (p:Emp)-[r:HELD]->(x) WHERE r.vf <= 3 AND r.vt >= 3 RETURN count(*) AS c";
        let plan = optimize(crate::gql::parse(q).unwrap());
        assert!(
            has_interval_expand(&plan),
            "optimizer did not fuse: {plan:?}"
        );
        // scan fallback (no interval index yet)
        assert!(matches!(run(&plan, &st).rows[0][0], Value::Num(x) if x == 3.0));
        // index seek (same plan, index present)
        st.create_interval_index("vf", "vt");
        assert!(matches!(run(&plan, &st).rows[0][0], Value::Num(x) if x == 3.0));

        // Row-level equivalence: the matching intervals' vf are {1,2,3}, seek == scan.
        let rq = "MATCH (p:Emp)-[r:HELD]->(x) WHERE r.vf <= 3 AND r.vt >= 3 RETURN r.vf AS f";
        let rplan = optimize(crate::gql::parse(rq).unwrap());
        let mut seek: Vec<String> = names_of(&run(&rplan, &st), 0);
        seek.sort();
        let scan_only = interval_store(); // fresh, no index
        let mut scan: Vec<String> = names_of(&run(&rplan, &scan_only), 0);
        scan.sort();
        assert_eq!(seek, scan);
        // vf of the matching intervals ([1,3],[2,4],[3,5]) — `names_of` renders a
        // Num via its debug form.
        assert_eq!(seek, vec!["Num(1.0)", "Num(2.0)", "Num(3.0)"]);
    }

    /// Grouping by an EDGE property counts per distinct edge-prop value — the
    /// bound edge sits at slot W and the endpoint node at W+1, so the count
    /// fast-path must not read the edge key as an (absent) node property. (The
    /// differential fuzzer found this bucketing every row under one NULL group.)
    #[test]
    fn group_by_edge_property_counts_per_value() {
        let mut b = Builder::default();
        let x = b.node(&["N"], &[]);
        let y = b.node(&["N"], &[]);
        let z = b.node(&["N"], &[]);
        b.edge(x, y, "R");
        b.edge(x, z, "R");
        b.edge(y, z, "R");
        let mut store = b.build();
        // Set weights: two edges w=2, one w=7 (eids 0,1,2 in insertion order).
        store.set_edge_prop(0, "w", n(2.0));
        store.set_edge_prop(1, "w", n(2.0));
        store.set_edge_prop(2, "w", n(7.0));
        let plan =
            crate::gql::parse("MATCH (a:N)-[r:R]->(b) RETURN r.w AS w, count(*) AS c").unwrap();
        // Group {2.0 → 2 edges, 7.0 → 1 edge}, order-independent.
        let mut got: Vec<(String, f64)> = run(&plan, &store)
            .rows
            .iter()
            .map(|row| (format!("{:?}", row[0]), num(&row[1])))
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            got,
            vec![("Num(2.0)".to_string(), 2.0), ("Num(7.0)".to_string(), 1.0)]
        );
    }

    /// K4: computed NaN/Inf are KEPT in the result value (matching lenke-core, so a
    /// caller can detect the signal), and coerced to null only at JSON egress.
    #[test]
    fn nan_and_inf_kept_in_results_coerced_at_egress() {
        let mut b = Builder::default();
        b.node(&["N"], &[("a", n(-4.0))]);
        let store = b.build();
        let val = |e: &str| {
            let q = format!("MATCH (x:N) RETURN {e} AS v");
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
        };
        assert!(matches!(val("sqrt(x.a)"), Value::Num(y) if y.is_nan())); // sqrt(-4) → NaN kept
        assert!(matches!(val("sqrt(x.a) + 1"), Value::Num(y) if y.is_nan())); // NaN propagates
        assert!(matches!(val("power(10, 400)"), Value::Num(y) if y.is_infinite())); // overflow → Inf
                                                                                    // But the JSON egress renders both as null (no JSON form for NaN/Inf).
        let ndjson = crate::ndjson::to_ndjson(&store);
        assert!(!ndjson.contains("NaN") && !ndjson.to_lowercase().contains("inf"));
    }

    /// Newly added scalar functions (K6 casts, K8 nullif, K9 math/constants,
    /// K5 size-on-string) match hand-computed values. One node with a=4, b="Carol".
    #[test]
    fn added_scalar_functions() {
        let mut b = Builder::default();
        b.node(&["N"], &[("a", n(4.0)), ("b", s("Carol"))]);
        let store = b.build();
        let val = |e: &str| -> Value {
            let q = format!("MATCH (n:N) RETURN {e} AS v");
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
        };
        let num = |e: &str| match val(e) {
            Value::Num(x) => x,
            o => panic!("{e} → {o:?}"),
        };
        // constants + math (native libm, matching core)
        assert!((num("pi()") - std::f64::consts::PI).abs() < 1e-12);
        assert!((num("e()") - std::f64::consts::E).abs() < 1e-12);
        assert_eq!(num("power(2, 10)"), 1024.0);
        assert_eq!(num("log(2, 8)"), 3.0); // log base 2 of 8 = ln(8)/ln(2)
        assert_eq!(num("mod(7, 3)"), 1.0);
        assert!((num("ln(e())") - 1.0).abs() < 1e-12);
        assert_eq!(num("degrees(pi())").round(), 180.0);
        // casts (NULL on a non-convertible input; a BOOLEAN converts to 1/0)
        assert_eq!(num("to_integer('7')"), 7.0);
        assert_eq!(num("to_integer(4.9)"), 4.0);
        assert_eq!(num("to_float('2.5')"), 2.5);
        assert!(matches!(val("to_string(n.a)"), Value::Str(x) if &*x == "4"));
        assert!(matches!(val("to_boolean('true')"), Value::Bool(true)));
        assert!(matches!(val("to_boolean(0)"), Value::Bool(false)));
        assert!(val("to_integer('nope')").is_null());
        assert_eq!(num("to_integer(true)"), 1.0); // explicit conversion coerces bool → 1/0
                                                  // nullif
        assert!(val("nullif(n.a, 4)").is_null());
        assert_eq!(num("nullif(n.a, 5)"), 4.0);
        // size / char_length on a string (K5)
        assert_eq!(num("size(n.b)"), 5.0);
        // `cardinality` is the ISO/SQL alias for `size` (a reserved word AND a function).
        assert_eq!(num("cardinality(n.b)"), 5.0);
        assert_eq!(num("cardinality([1, 2, 3])"), 3.0);
        assert_eq!(num("char_length(n.b)"), 5.0);
    }

    /// Subscript `base[index]` (ISO 0-based) over a list literal, record and map:
    /// in-range element, out-of-range / negative / non-integer → NULL, null-safe.
    #[test]
    fn subscript_list_record_map() {
        let mut b = Builder::default();
        b.node(&["N"], &[("z", n(1.0))]);
        let store = b.build();
        let num = |e: &str| -> f64 {
            let q = format!("MATCH (x:N) RETURN {e} AS v");
            match run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0] {
                Value::Num(x) => x,
                ref o => panic!("{e} → {o:?}"),
            }
        };
        let isnull = |e: &str| -> bool {
            let q = format!("MATCH (x:N) RETURN {e} AS v");
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].is_null()
        };
        assert_eq!(num("[10,20,30][0]"), 10.0);
        assert_eq!(num("[10,20,30][2]"), 30.0);
        assert!(isnull("[10,20,30][9]")); // out of range
        assert!(isnull("[10,20,30][-1]")); // negative
        assert!(isnull("[10,20,30][1.5]")); // non-integer
        assert_eq!(num("{a:1,b:2}['b']"), 2.0); // record field by string key
        assert!(isnull("{a:1,b:2}['zzz']")); // missing field
    }

    /// `edges(p)[i]` / `nodes(p)[i]` keep element typing so a following `.prop`
    /// resolves the edge/node property. Path n0 -R(w=5)-> n1 -R(w=7)-> n2.
    #[test]
    fn subscript_path_element_property() {
        let nd = concat!(
            "{\"id\":\"n0\",\"labels\":[\"N\"],\"props\":{\"id\":\"n0\"}}\n",
            "{\"id\":\"n1\",\"labels\":[\"N\"],\"props\":{\"id\":\"n1\"}}\n",
            "{\"id\":\"n2\",\"labels\":[\"N\"],\"props\":{\"id\":\"n2\"}}\n",
            "{\"id\":\"e1\",\"from\":\"n0\",\"to\":\"n1\",\"type\":\"R\",\"props\":{\"w\":5}}\n",
            "{\"id\":\"e2\",\"from\":\"n1\",\"to\":\"n2\",\"type\":\"R\",\"props\":{\"w\":7}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let val = |e: &str| -> Value {
            let q = format!(
                "MATCH p = ANY SHORTEST (a:N {{id:'n0'}})-[:R]->*(b:N {{id:'n2'}}) RETURN {e} AS v"
            );
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
        };
        assert!(matches!(val("edges(p)[0].w"), Value::Num(x) if x == 5.0)); // first edge property
        assert!(matches!(val("edges(p)[1].w"), Value::Num(x) if x == 7.0)); // second edge property
        assert!(val("edges(p)[9].w").is_null()); // out-of-range edge → NULL prop
        assert!(matches!(val("nodes(p)[2].id"), Value::Str(x) if &*x == "n2"));
        assert!(val("nodes(p)[0].nope").is_null()); // missing node property
        assert!(matches!(
            val("edges(p)[1].w > edges(p)[0].w"),
            Value::Bool(true)
        ));
    }

    /// `VALUE { MATCH (a)-[:R]->(b) RETURN count(*) }` is a correlated count subquery
    /// (a degree), lowering to the same result as `COUNT { (a)-[:R]->(b) }`.
    #[test]
    fn value_count_subquery() {
        let nd = concat!(
            "{\"id\":\"dave\",\"labels\":[\"P\"],\"props\":{\"id\":\"dave\"}}\n",
            "{\"id\":\"carol\",\"labels\":[\"P\"],\"props\":{\"id\":\"carol\"}}\n",
            "{\"id\":\"x\",\"labels\":[\"P\"],\"props\":{\"id\":\"x\"}}\n",
            "{\"id\":\"y\",\"labels\":[\"P\"],\"props\":{\"id\":\"y\"}}\n",
            "{\"from\":\"dave\",\"to\":\"x\",\"labels\":[\"KNOWS\"],\"props\":{}}\n",
            "{\"from\":\"dave\",\"to\":\"y\",\"labels\":[\"KNOWS\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let deg = |id: &str| -> f64 {
            let q = format!(
                "MATCH (a:P) WHERE a.id='{id}' RETURN VALUE {{ MATCH (a)-[:KNOWS]->(b) RETURN count(*) }} AS deg"
            );
            let plan = crate::opt::optimize_indexed(crate::gql::parse(&q).unwrap(), &store);
            match run(&plan, &store).rows[0][0] {
                Value::Num(x) => x,
                ref o => panic!("{o:?}"),
            }
        };
        assert_eq!(deg("dave"), 2.0); // dave knows x and y
        assert_eq!(deg("carol"), 0.0); // carol knows no one
    }

    /// Multi-label edges: an edge's type is its FIRST label; the rest are secondary
    /// labels a `-[:label]->` hop must still match. `a-[:X,:Y]->b`, `a-[:Y]->c`.
    #[test]
    fn multi_label_edge_matching() {
        // a -r0[X,Y]-> b ; a -r1[Y]-> c ; b -r2[Z,Y]-> c
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"r0\",\"from\":\"a\",\"to\":\"b\",\"labels\":[\"X\",\"Y\"],\"props\":{}}\n",
            "{\"id\":\"r1\",\"from\":\"a\",\"to\":\"c\",\"labels\":[\"Y\"],\"props\":{}}\n",
            "{\"id\":\"r2\",\"from\":\"b\",\"to\":\"c\",\"labels\":[\"Z\",\"Y\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        assert!(store.has_multi_label_edges());
        let ids = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        // `:Y` reaches every edge (all three carry Y, two only as a secondary label).
        assert_eq!(
            ids("MATCH (a:N)-[:Y]->(b) RETURN b.id AS x"),
            vec!["b", "c", "c"]
        );
        // `:X` only r0 (its primary), `:Z` only r2 (its primary).
        assert_eq!(ids("MATCH (a:N)-[:X]->(b) RETURN b.id AS x"), vec!["b"]);
        assert_eq!(ids("MATCH (a:N)-[:Z]->(b) RETURN b.id AS x"), vec!["c"]);
        // A var-length `:Y` hop crosses secondary-label edges too: a-Y->b-Y->c.
        assert!(
            ids("MATCH (a:N {id:'a'})-[:Y]->{2}(b) RETURN b.id AS x").contains(&"c".to_string())
        );
        // type(edge) is the FIRST label, not a secondary one.
        let ty = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        assert_eq!(
            ty("MATCH (a:N)-[e:Y]->(b) RETURN type(e) AS t"),
            vec!["X", "Y", "Z"]
        );
    }

    /// Edge-label NEGATION `-[:!T]->` matches any edge whose type is NOT `T` (the
    /// complement of the named types), and `:!(A|B)` negates a disjunction.
    #[test]
    fn edge_label_negation() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"P\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"P\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"s\",\"labels\":[\"P\"],\"props\":{\"id\":\"s\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"KNOWS\"],\"props\":{}}\n",
            "{\"from\":\"a\",\"to\":\"c\",\"labels\":[\"CREATED\"],\"props\":{}}\n",
            "{\"from\":\"a\",\"to\":\"s\",\"labels\":[\"LIKES\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let ids = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        // NOT CREATED → the KNOWS and LIKES targets.
        assert_eq!(
            ids("MATCH (a:P {id:'a'})-[:!CREATED]->(x) RETURN x.id AS x"),
            vec!["b", "s"]
        );
        // NOT (CREATED|LIKES) → only the KNOWS target.
        assert_eq!(
            ids("MATCH (a:P {id:'a'})-[:!(CREATED|LIKES)]->(x) RETURN x.id AS x"),
            vec!["b"]
        );
        // A negated unknown type excludes nothing → every out-edge.
        assert_eq!(
            ids("MATCH (a:P {id:'a'})-[:!NOSUCH]->(x) RETURN x.id AS x"),
            vec!["b", "c", "s"]
        );
    }

    /// Inline edge properties on a plain var-length hop filter every edge on the
    /// path. a-e(10)->b-e(20)->c-e(5)->d; only b->c has amt 20.
    #[test]
    fn var_length_inline_edge_props() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":20.0}}\n",
            "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{\"amt\":5.0}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let ids = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        // From a, no outgoing amt=20 edge → no path.
        assert!(ids("MATCH (a:N {id:'a'})-[:R {amt:20.0}]->{1,3}(x) RETURN x.id AS id").is_empty());
        // From b, b->c has amt 20 → x = c (c->d is amt 5, excluded).
        assert_eq!(
            ids("MATCH (b:N {id:'b'})-[:R {amt:20.0}]->{1,3}(x) RETURN x.id AS id"),
            vec!["c"]
        );
    }

    /// A per-hop edge WHERE on a plain var-length hop filters each hop's edge.
    /// a-e(20)->b-e(5)->c: e.amt>=10 admits only a->b.
    #[test]
    fn plain_var_length_per_hop_where() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":20.0}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":5.0}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let ids = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        // e.amt >= 10 blocks b->c → only a->b reaches b.
        assert_eq!(
            ids("MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 10]->{1,3}(x) RETURN x.id AS id"),
            vec!["b"]
        );
        // e.amt >= 1 admits all → b, c.
        assert_eq!(
            ids("MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 1]->{1,3}(x) RETURN x.id AS id"),
            vec!["b", "c"]
        );
    }

    /// A per-hop edge WHERE may also reference the hop's SOURCE variable: `(a)-[e
    /// WHERE a.id = … AND e.amt >= …]->{1,3}(x)`. The anchor `a` maps to the path
    /// source at eval time, so a true condition admits the walk and a false one
    /// blocks every path.
    #[test]
    fn per_hop_where_references_outer_source() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":20.0}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":20.0}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let ids = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        // a.id = 'a' holds → both hops admitted (b, c).
        assert_eq!(
            ids("MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 10 AND a.id = 'a']->{1,3}(x) RETURN x.id AS id"),
            vec!["b", "c"]
        );
        // a.id = 'zzz' is false → no path survives.
        assert!(
            ids("MATCH (a:N {id:'a'})-[e:R WHERE a.id = 'zzz']->{1,3}(x) RETURN x.id AS id")
                .is_empty()
        );
    }

    /// Graph-element predicates: IS DIRECTED, IS SOURCE/DESTINATION OF, ALL_DIFFERENT,
    /// SAME — three-valued over element identity (a null operand → NULL).
    #[test]
    fn graph_element_predicates() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let row = |q: &str| -> Vec<Value> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            run(&plan, &store).rows[0].to_vec()
        };
        let r = row(
            "MATCH (a:N {id:'a'})-[e:R]->(b:N {id:'b'}) RETURN e IS DIRECTED AS d, \
             a IS SOURCE OF e AS asrc, b IS DESTINATION OF e AS bdst, b IS SOURCE OF e AS bsrc, \
             ALL_DIFFERENT(a, b) AS diff, SAME(a, a) AS saa, SAME(a, b) AS sab",
        );
        assert!(matches!(r[0], Value::Bool(true))); // e IS DIRECTED
        assert!(matches!(r[1], Value::Bool(true))); // a IS SOURCE OF e
        assert!(matches!(r[2], Value::Bool(true))); // b IS DESTINATION OF e
        assert!(matches!(r[3], Value::Bool(false))); // b IS SOURCE OF e
        assert!(matches!(r[4], Value::Bool(true))); // ALL_DIFFERENT(a,b)
        assert!(matches!(r[5], Value::Bool(true))); // SAME(a,a)
        assert!(matches!(r[6], Value::Bool(false))); // SAME(a,b)
                                                     // Three-valued: a null element → NULL.
        let r = row("MATCH (a:N {id:'a'}) OPTIONAL MATCH (a)-[:NOSUCH]->(m) \
             RETURN m IS DIRECTED AS d, ALL_DIFFERENT(a, m) AS ad");
        assert!(r[0].is_null());
        assert!(r[1].is_null());
    }

    /// Bare ALL/ANY selectors: ALL is the default (every path — a duplicate endpoint
    /// per path), ANY keeps one per endpoint (dedup). Diamond a->b->d, a->c->d.
    #[test]
    fn bare_all_any_selectors() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"a\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"b\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let ids = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        // ALL: d reached by two 2-hop paths → d appears twice; b, c once each.
        assert_eq!(
            ids("MATCH ALL (a:N {id:'a'})-[:R]->{1,2}(x) RETURN x.id AS id"),
            vec!["b", "c", "d", "d"]
        );
        // ANY: one per endpoint → b, c, d once each.
        assert_eq!(
            ids("MATCH ANY (a:N {id:'a'})-[:R]->{1,2}(x) RETURN x.id AS id"),
            vec!["b", "c", "d"]
        );
    }

    /// FOR..IN list unwind: literal list, ordinal (1-based ORDINALITY / 0-based
    /// OFFSET), null/empty → no rows, a scalar singleton, and multiplying a MATCH.
    #[test]
    fn for_in_unwind() {
        let mut b = Builder::default();
        b.node(&["P"], &[("name", s("marko"))]);
        let store = b.build();
        let nums = |q: &str| -> Vec<f64> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            run(&plan, &store)
                .rows
                .iter()
                .map(|r| match r[0] {
                    Value::Num(x) => x,
                    ref o => panic!("{o:?}"),
                })
                .collect()
        };
        assert_eq!(nums("FOR x IN [1, 2, 3] RETURN x"), vec![1.0, 2.0, 3.0]);
        // ORDINALITY is 1-based, OFFSET 0-based.
        assert_eq!(
            nums("FOR x IN ['a','b'] WITH ORDINALITY i RETURN i"),
            vec![1.0, 2.0]
        );
        assert_eq!(
            nums("FOR x IN ['a','b'] WITH OFFSET i RETURN i"),
            vec![0.0, 1.0]
        );
        // null and empty list → no rows; a non-list scalar → one row.
        assert_eq!(nums("FOR x IN null RETURN x").len(), 0);
        assert_eq!(nums("FOR x IN [] RETURN x").len(), 0);
        assert_eq!(nums("FOR x IN 5 RETURN x"), vec![5.0]);
        // Multiplies a prior MATCH (one row per (match, element)).
        let plan = crate::opt::optimize_indexed(
            crate::gql::parse("MATCH (p:P) FOR t IN ['x','y'] RETURN t").unwrap(),
            &store,
        );
        assert_eq!(run(&plan, &store).rows.len(), 2);
    }

    /// A FOR-driven fresh-variable `OPTIONAL MATCH (p:Label {k: expr})` is a left-outer
    /// correlated scan: each unwound name finds the matching node (its age), or a NULL
    /// node when none matches.
    #[test]
    fn for_driven_optional_scan() {
        let mut b = Builder::default();
        b.node(&["Person"], &[("name", s("josh")), ("age", n(32.0))]);
        b.node(&["Person"], &[("name", s("marko")), ("age", n(29.0))]);
        let store = b.build();
        let plan = crate::opt::optimize_indexed(
            crate::gql::parse(
                "FOR name IN ['josh', 'nobody'] \
                 OPTIONAL MATCH (p:Person {name: name}) RETURN name, p.age",
            )
            .unwrap(),
            &store,
        );
        let rows: Vec<(String, Value)> = run(&plan, &store)
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(nm) => (nm.to_string(), r[1].clone()),
                o => panic!("{o:?}"),
            })
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, "josh");
        assert!(matches!(rows[0].1, Value::Num(x) if x == 32.0));
        assert_eq!(rows[1].0, "nobody");
        assert!(rows[1].1.is_null());
    }

    /// A single-outer-rep endpoint-only nested group `( ()-[:R]->{1,3}() ){1} (t)`
    /// desugars to a var-length {1,3}. Chain a->b->c->d.
    #[test]
    fn nested_endpoint_only_single_rep() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let plan = crate::opt::optimize_indexed(
            crate::gql::parse("MATCH (s:N {id:'a'}) ( ()-[:R]->{1,3}() ){1} (t) RETURN t.id AS id")
                .unwrap(),
            &store,
        );
        let mut v = names_of(&run(&plan, &store), 0);
        v.sort();
        assert_eq!(v, vec!["b", "c", "d"]); // reachable in 1..3 hops
                                            // A MULTI-repetition endpoint-only nested group now enumerates each
                                            // rep-decomposition (`Plan::NestedGroup`): `( ()-[:R]->{1,2}() ){2}` from a on
                                            // the chain a->b->c->d = 2 outer reps, each 1-2 hops (trail). Endpoints (with
                                            // multiplicity, one row per decomposition): 2+2=c, 2+... only c and d reach.
                                            // a->b then b->c (c), a->b then b->c->d (d), a->b->c then c->d (d).
        let plan2 = crate::opt::optimize_indexed(
            crate::gql::parse("MATCH (s:N {id:'a'}) ( ()-[:R]->{1,2}() ){2} (t) RETURN t.id AS id")
                .unwrap(),
            &store,
        );
        let mut v2 = names_of(&run(&plan2, &store), 0);
        v2.sort();
        assert_eq!(v2, vec!["c", "d", "d"]);
    }

    /// A repeated pattern variable on a var-length landing is an equality join: an
    /// EXISTS correlated on BOTH anchors `EXISTS { (a)-[:R]->+(b) }`, and a cycle
    /// `(a)-[:R]->{1,3}(a)`. Chain a->b->c (no cycle back to a).
    #[test]
    fn repeated_variable_landing_equality() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let rows = |q: &str| -> usize {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            run(&plan, &store).rows.len()
        };
        // a reaches c → EXISTS true → 1 row.
        assert_eq!(
            rows("MATCH (a:N {id:'a'}), (b:N {id:'c'}) WHERE EXISTS { MATCH (a)-[:R]->+(b) } RETURN 1 AS x"),
            1
        );
        // a does NOT reach a (no cycle) → EXISTS false → 0 rows.
        assert_eq!(
            rows("MATCH (a:N {id:'a'}), (b:N {id:'a'}) WHERE EXISTS { MATCH (a)-[:R]->+(b) } RETURN 1 AS x"),
            0
        );
        // A named cycle `(a)…(a)`: a can't return to a → no path.
        assert_eq!(
            rows("MATCH p = SIMPLE (a:N {id:'a'})-[:R]->{1,3}(a) RETURN path_length(p) AS len"),
            0
        );
    }

    /// An uncorrelated VALUE subquery runs a self-contained body once: a constant
    /// (`VALUE { RETURN 1+2 }`) or a global aggregate (`VALUE { MATCH (n) RETURN
    /// count(*) }`).
    #[test]
    fn uncorrelated_value_subquery() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"Person\"],\"props\":{}}\n",
            "{\"id\":\"b\",\"labels\":[\"Person\"],\"props\":{}}\n",
            "{\"id\":\"c\",\"labels\":[\"Person\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let num = |q: &str| -> f64 {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            match run(&plan, &store).rows[0][0] {
                Value::Num(x) => x,
                ref o => panic!("{o:?}"),
            }
        };
        assert_eq!(num("RETURN VALUE { RETURN 1 + 2 } AS v"), 3.0);
        assert_eq!(
            num("RETURN VALUE { MATCH (n:Person) RETURN count(*) } AS c"),
            3.0
        );
    }

    /// An uncorrelated multi-pattern EXISTS `EXISTS { MATCH (x:N) MATCH (y:M) }` is a
    /// self-contained cross-join existence check, run once and broadcast.
    #[test]
    fn uncorrelated_multi_match_exists() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"M\"],\"props\":{\"id\":\"b\"}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let b = |q: &str| -> bool {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            matches!(run(&plan, &store).rows[0][0], Value::Bool(true))
        };
        // N and M both non-empty → true.
        assert!(b("RETURN EXISTS { MATCH (x:N) MATCH (y:M) } AS e"));
        // Z is empty → the cross-join is empty → false.
        assert!(!b("RETURN EXISTS { MATCH (x:N) MATCH (y:Z) } AS e"));
        // Per-clause WHERE: x.id='a' and y.id='b' both match → true.
        assert!(b(
            "RETURN EXISTS { MATCH (x:N) WHERE x.id='a' MATCH (y:M) WHERE y.id='b' } AS e"
        ));
        // y.id='nope' matches nothing → false.
        assert!(!b(
            "RETURN EXISTS { MATCH (x:N) WHERE x.id='a' MATCH (y:M) WHERE y.id='nope' } AS e"
        ));
    }

    /// A correlated scalar VALUE subquery returns the body's single value per outer
    /// row (NULL if empty), and ERRORS if the body matches more than one row.
    #[test]
    fn scalar_value_subquery() {
        let nd = concat!(
            "{\"id\":\"alice\",\"labels\":[\"Person\"],\"props\":{\"id\":\"alice\",\"name\":\"Alice\"}}\n",
            "{\"id\":\"carol\",\"labels\":[\"Person\"],\"props\":{\"id\":\"carol\",\"name\":\"Carol\"}}\n",
            "{\"id\":\"dave\",\"labels\":[\"Person\"],\"props\":{\"id\":\"dave\",\"name\":\"Dave\"}}\n",
            "{\"id\":\"bob\",\"labels\":[\"Person\"],\"props\":{\"id\":\"bob\",\"name\":\"Bob\"}}\n",
            "{\"id\":\"erin\",\"labels\":[\"Person\"],\"props\":{\"id\":\"erin\",\"name\":\"Erin\"}}\n",
            "{\"from\":\"alice\",\"to\":\"bob\",\"labels\":[\"KNOWS\"],\"props\":{}}\n",
            "{\"from\":\"dave\",\"to\":\"bob\",\"labels\":[\"KNOWS\"],\"props\":{}}\n",
            "{\"from\":\"dave\",\"to\":\"erin\",\"labels\":[\"KNOWS\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let one = |id: &str| -> Value {
            let q = format!(
                "MATCH (a:Person) WHERE a.id='{id}' RETURN VALUE {{ MATCH (a)-[:KNOWS]->(b) RETURN b.name }} AS f"
            );
            let plan = crate::opt::optimize_indexed(crate::gql::parse(&q).unwrap(), &store);
            run(&plan, &store).rows[0][0].clone()
        };
        assert!(matches!(one("alice"), Value::Str(s) if &*s == "Bob")); // one friend
        assert!(one("carol").is_null()); // no friend → NULL
                                         // dave knows two → the subquery returns >1 row → execute errors.
        let q = "MATCH (a:Person) WHERE a.id='dave' RETURN VALUE { MATCH (a)-[:KNOWS]->(b) RETURN b.name } AS f";
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        assert!(try_run(&plan, &store).is_err());
    }

    /// OPTIONAL MATCH binding an edge variable `(a)-[f:R]->(b)` binds the edge slot
    /// too (left-outer: null edge + null node on a miss). a->b->c, c has no out edge.
    #[test]
    fn optional_match_binds_edge() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"w\":7.0}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"w\":9.0}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        // For each N, OPTIONAL MATCH one outgoing R edge; RETURN the node id + f.w.
        // a->b (w7), b->c (w9), c has none → f.w NULL.
        let q =
            "MATCH (n:N) OPTIONAL MATCH (n)-[f:R]->(u) RETURN n.id AS id, f.w AS w ORDER BY n.id";
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let out = run(&plan, &store);
        let got: Vec<(String, Value)> = out
            .rows
            .iter()
            .map(|r| {
                let id = match &r[0] {
                    Value::Str(s) => s.to_string(),
                    o => format!("{o:?}"),
                };
                (id, r[1].clone())
            })
            .collect();
        assert_eq!(got.len(), 3);
        assert!(matches!(&got[0], (id, Value::Num(w)) if id == "a" && *w == 7.0));
        assert!(matches!(&got[1], (id, Value::Num(w)) if id == "b" && *w == 9.0));
        assert!(matches!(&got[2], (id, v) if id == "c" && v.is_null())); // no edge → f null
    }

    /// A per-repetition WHERE on a MULTI-HOP unit references every edge of the rep
    /// (e1 AND e2), checked at the rep boundary. Chain a-b-c-d-e, all amt 10.
    #[test]
    fn multi_hop_group_per_rep_where() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
            "{\"id\":\"e\",\"labels\":[\"N\"],\"props\":{\"id\":\"e\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}\n",
            "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}\n",
            "{\"from\":\"d\",\"to\":\"e\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let ids = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        // e2.amt <= e1.amt (10<=10) holds → 1 rep (t=c), 2 reps (t=e).
        assert_eq!(
            ids("MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y) WHERE e2.amt <= e1.amt){1,2} (t) RETURN t.id AS id"),
            vec!["c", "e"]
        );
        // e2.amt < e1.amt (10<10) fails every rep → no path.
        assert!(
            ids("MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y) WHERE e2.amt < e1.amt){1,2} (t) RETURN t.id AS id")
                .is_empty()
        );
    }

    /// NESTED subpath groups (`Plan::NestedGroup`): group variables materialize as
    /// (nested) lists — one list level per enclosing quantifier. On the triangle
    /// a->b->c->a: family 4 `( (x)-[e:R]->{1,2}(y) ){1,2}` binds x/y once per OUTER rep
    /// (depth 1) and e as a list-of-lists (depth 2); family 3 `( ((x)-[e]->(y)){1,2}
    /// ){1,2}` binds x as a list-of-lists (depth 2).
    #[test]
    fn nested_subpath_groups() {
        // Chain a->b->c->d->e (ids 0..4).
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":0}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":1}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":2}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":3}}\n",
            "{\"id\":\"e\",\"labels\":[\"N\"],\"props\":{\"id\":4}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"d\",\"to\":\"e\",\"labels\":[\"R\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let rows = |q: &str| -> Vec<Vec<f64>> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v: Vec<Vec<f64>> = run(&plan, &store)
                .rows
                .iter()
                .map(|r| {
                    r.iter()
                        .map(|c| match c {
                            Value::Num(x) => *x,
                            o => panic!("{o:?}"),
                        })
                        .collect()
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        // Family 4 from a: the 6 trail decompositions of outer{1,2}×inner{1,2}.
        // (tid, size(x), size(e)) — size(x)/size(e) = the outer rep count.
        assert_eq!(
            rows(
                "MATCH (s:N {id:0}) ( (x)-[e:R]->{1,2}(y) ){1,2} (t) \
                  RETURN t.id AS tid, size(x) AS nx, size(e) AS ne"
            ),
            vec![
                vec![1.0, 1.0, 1.0],
                vec![2.0, 1.0, 1.0],
                vec![2.0, 2.0, 2.0],
                vec![3.0, 2.0, 2.0],
                vec![3.0, 2.0, 2.0],
                vec![4.0, 2.0, 2.0],
            ]
        );
        // Family 3 `( ((x)-[e]->(y)){2,2} ){2} (t)` from a: exactly one match — 2 outer
        // reps of 2 inner hops = the 4-hop trail a->b->c->d->e, endpoint e(4). x is
        // depth-2: size(x)=2 (outer), size(x[0])=2 (inner), x[0][0]=a(0).
        assert_eq!(
            rows(
                "MATCH (s:N {id:0}) ( ((x)-[e:R]->(y)){2,2} ){2} (t) \
                  RETURN t.id AS tid, size(x) AS nx, size(x[0]) AS nx0, x[0][0].id AS a00"
            ),
            vec![vec![4.0, 2.0, 2.0, 0.0]]
        );
    }

    /// A MULTI-HOP group unit `((x)-[e1]->(m)-[e2]->(y)){2}` binds each inner var to
    /// a list strided by the unit hop count k; the endpoint lands at a rep boundary.
    /// Chain a-11->b-22->c-33->d-44->e. 2 reps of 2 hops → t=e.
    #[test]
    fn repeat_group_multi_hop_unit() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
            "{\"id\":\"e\",\"labels\":[\"N\"],\"props\":{\"id\":\"e\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":11.0}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":22.0}}\n",
            "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{\"amt\":33.0}}\n",
            "{\"from\":\"d\",\"to\":\"e\",\"labels\":[\"R\"],\"props\":{\"amt\":44.0}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let q = "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y)){2} (t) \
                 RETURN t.id AS tid, x[0].id AS x0, x[1].id AS x1, m[1].id AS m1, \
                 y[1].id AS y1, e1[0].amt AS p0, e2[1].amt AS q1";
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1);
        let r = &out.rows[0];
        assert!(matches!(&r[0], Value::Str(s) if &**s == "e")); // t = e
        assert!(matches!(&r[1], Value::Str(s) if &**s == "a")); // x[0] = a
        assert!(matches!(&r[2], Value::Str(s) if &**s == "c")); // x[1] = c
        assert!(matches!(&r[3], Value::Str(s) if &**s == "d")); // m[1] = d
        assert!(matches!(&r[4], Value::Str(s) if &**s == "e")); // y[1] = e
        assert!(matches!(r[5], Value::Num(x) if x == 11.0)); // e1[0].amt
        assert!(matches!(r[6], Value::Num(x) if x == 44.0)); // e2[1].amt
    }

    /// A per-repetition WHERE prunes each hop by the rep's scalar x/e/y. Path
    /// a-e1(30)->b-e2(20)->c-e3(10)->d; bals a=100,b=200,c=5,d=200.
    #[test]
    fn repeat_group_per_rep_where() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\",\"bal\":100.0}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\",\"bal\":200.0}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\",\"bal\":5.0}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\",\"bal\":200.0}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":30.0}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":20.0}}\n",
            "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let ids = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        // e.amt >= 1 holds for every edge → reach b, c, d.
        assert_eq!(
            ids(
                "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE e.amt >= 1){1,3} (t) RETURN t.id AS id"
            ),
            vec!["b", "c", "d"]
        );
        // e.amt <= x.bal fails at c->d (10 <= 5 false) → only b, c.
        assert_eq!(
            ids("MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE e.amt <= x.bal){1,3} (t) RETURN t.id AS id"),
            vec!["b", "c"]
        );
        // y.bal >= 100 fails when y=c (bal 5) → only b (a->b).
        assert_eq!(
            ids("MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE y.bal >= 100){1,3} (t) RETURN t.id AS id"),
            vec!["b"]
        );
    }

    /// A per-hop edge WHERE in a shortest path filters which edges may be traversed.
    /// a-e1(w1)->b, a-e2(w10)->c, c-e3(w10)->b. With w>5, e1 is blocked → a->c->b (2).
    #[test]
    fn shortest_path_per_hop_edge_where() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"w\":1.0}}\n",
            "{\"from\":\"a\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"w\":10.0}}\n",
            "{\"from\":\"c\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"w\":10.0}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let lens = |q: &str| -> Vec<f64> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            run(&plan, &store)
                .rows
                .iter()
                .map(|r| match r[0] {
                    Value::Num(x) => x,
                    ref o => panic!("{o:?}"),
                })
                .collect()
        };
        // e.w > 5 blocks a->b (w1); shortest a->b is a->c->b, length 2.
        assert_eq!(
            lens("MATCH p = ANY SHORTEST (a:N {id:'a'})-[e:R WHERE e.w > 5]->*(b:N {id:'b'}) RETURN path_length(p) AS len"),
            vec![2.0]
        );
        // e.w > 100 blocks every edge → b unreachable.
        assert!(
            lens("MATCH p = ANY SHORTEST (a:N {id:'a'})-[e:R WHERE e.w > 100]->*(b:N {id:'b'}) RETURN path_length(p) AS len")
                .is_empty()
        );
    }

    /// SHORTEST k (k>=2) keeps the k shortest trails per endpoint by (length,
    /// discovery); GROUP keeps every trail in the k smallest distinct lengths.
    /// a->d (1), a->b->d (2), a->c->d (2).
    #[test]
    fn shortest_k_selector() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
            "{\"from\":\"a\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"b\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"a\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let lens = |q: &str| -> Vec<f64> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v: Vec<f64> = run(&plan, &store)
                .rows
                .iter()
                .map(|r| match r[0] {
                    Value::Num(x) => x,
                    ref o => panic!("{o:?}"),
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        // SHORTEST 2 → the two shortest: len 1 and one len 2.
        assert_eq!(
            lens("MATCH p = SHORTEST 2 (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len"),
            vec![1.0, 2.0]
        );
        // SHORTEST 2 GROUP → all trails in the 2 smallest lengths (1 and 2): 1,2,2.
        assert_eq!(
            lens("MATCH p = SHORTEST 2 GROUP (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len"),
            vec![1.0, 2.0, 2.0]
        );
        // SHORTEST 10 clamps to the 3 available.
        assert_eq!(
            lens("MATCH p = SHORTEST 10 (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len"),
            vec![1.0, 2.0, 2.0]
        );
    }

    /// A quantified subpath group binds its inner variables as GROUP lists: each
    /// becomes a list over the repetitions, with `size()` the hop count and `v[i]`
    /// a typed node/edge element so `x[i].prop` resolves. Path a-R(10)->b-R(20)->c.
    #[test]
    fn repeat_group_binds_group_variables() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":10}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":20}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let row = |q: &str| -> Vec<Value> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let out = run(&plan, &store);
            assert_eq!(out.rows.len(), 1, "{q}");
            out.rows[0].to_vec()
        };
        // {2}: t=c, size(e)=size(x)=size(y)=2, x[0]=a, y[1]=c, e[0].amt=10.
        let r = row(
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){2} (t) \
             RETURN t.id AS tid, size(e) AS ne, size(x) AS nx, x[0].id AS x0, y[1].id AS y1, e[0].amt AS e0",
        );
        assert!(matches!(&r[0], Value::Str(s) if &**s == "c")); // tid
        assert!(matches!(r[1], Value::Num(x) if x == 2.0)); // size(e)
        assert!(matches!(r[2], Value::Num(x) if x == 2.0)); // size(x)
        assert!(matches!(&r[3], Value::Str(s) if &**s == "a")); // x[0].id
        assert!(matches!(&r[4], Value::Str(s) if &**s == "c")); // y[1].id
        assert!(matches!(r[5], Value::Num(x) if x == 10.0)); // e[0].amt
                                                             // Anonymous endpoint: only the group vars are used.
        let r = row("MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){2} RETURN size(e) AS ne");
        assert!(matches!(r[0], Value::Num(x) if x == 2.0));
    }

    /// `IS TYPED RECORD { f :: TYPE [NOT NULL], … }` is a CLOSED record type: no
    /// extra fields, every declared field present-and-typed or absent-and-nullable,
    /// recursing into nested records. INTEGER requires an integral number.
    #[test]
    fn is_typed_closed_record() {
        let store =
            crate::ndjson::from_ndjson("{\"id\":\"1\",\"labels\":[\"X\"],\"props\":{}}").unwrap();
        let row = |q: &str| -> Vec<bool> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            run(&plan, &store).rows[0]
                .iter()
                .map(|v| matches!(v, Value::Bool(true)))
                .collect()
        };
        assert_eq!(
            row(
                "RETURN {a: 1, b: 'x'} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS a, \
                 {a: 1} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS b, \
                 {a: 1, b: 'x', c: 9} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS c, \
                 {a: 1.5} IS TYPED RECORD {a :: INTEGER} AS d, \
                 {a: 1.5} IS TYPED RECORD {a :: FLOAT} AS e"
            ),
            vec![true, true, false, false, true]
        );
        assert_eq!(
            row("RETURN {} IS TYPED RECORD {a :: INTEGER NOT NULL} AS a, \
                 {a: null} IS TYPED RECORD {a :: INTEGER NOT NULL} AS b, \
                 {geo: {lat: 1, lng: 2}} IS TYPED RECORD {geo :: RECORD {lat :: INTEGER, lng :: INTEGER}} AS c, \
                 {geo: {lat: 'x'}} IS TYPED RECORD {geo :: RECORD {lat :: INTEGER, lng :: INTEGER}} AS d"),
            vec![false, false, true, false]
        );
    }

    /// A group-variable EDGE list keeps its element typing across a `WITH … AS`
    /// rename: `WITH e AS hops` leaves `hops[i].amt` resolving the edge property, not
    /// NULL. (The parser remaps the edge-/node-list slot sets through the WITH.)
    #[test]
    fn group_variable_edge_typing_survives_with_rename() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":11}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":22}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let plan = crate::opt::optimize_indexed(
            crate::gql::parse(
                "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){2} (t) \
                 WITH e AS hops, t RETURN t.id AS tid, hops[1].amt AS amt2, hops[0].amt AS amt1",
            )
            .unwrap(),
            &store,
        );
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1);
        let r = out.rows[0].to_vec();
        assert!(matches!(&r[0], Value::Str(s) if &**s == "c"));
        assert!(matches!(r[1], Value::Num(x) if x == 22.0)); // hops[1].amt
        assert!(matches!(r[2], Value::Num(x) if x == 11.0)); // hops[0].amt
    }

    /// A standalone FILTER clause filters the working table, and repeated statement-
    /// position ORDER BY … LIMIT compose (page then re-page). n = 1,5,9.
    #[test]
    fn filter_clause_and_composed_paging() {
        let mut b = Builder::default();
        b.node(&["T"], &[("n", n(1.0))]);
        b.node(&["T"], &[("n", n(5.0))]);
        b.node(&["T"], &[("n", n(9.0))]);
        let store = b.build();
        let one = |q: &str| -> f64 {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            match run(&plan, &store).rows[0][0] {
                Value::Num(x) => x,
                ref o => panic!("{o:?}"),
            }
        };
        // FILTER keeps n>3 (5,9); ORDER BY n LIMIT 1 -> 5.
        assert_eq!(
            one("MATCH (t:T) FILTER t.n > 3 ORDER BY t.n LIMIT 1 RETURN t.n AS x"),
            5.0
        );
        // Page (asc, top 2 -> {1,5}) then re-page (desc, top 1 -> 5).
        assert_eq!(
            one("MATCH (t:T) ORDER BY t.n LIMIT 2 ORDER BY t.n DESC LIMIT 1 RETURN t.n AS x"),
            5.0
        );
    }

    /// An UNQUANTIFIED subpath group `(( pattern [WHERE p] ))` is a scoping paren:
    /// the inner pattern + trailing WHERE filter, no repetition. A NAMED path over
    /// one is rejected (core does). Fixture: Amy(25)->Bob(40), Bob(40)->Amy(25).
    #[test]
    fn unquantified_subpath_group() {
        let nd = concat!(
            "{\"id\":\"amy\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Amy\",\"age\":25}}\n",
            "{\"id\":\"bob\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Bob\",\"age\":40}}\n",
            "{\"from\":\"amy\",\"to\":\"bob\",\"labels\":[\"KNOWS\"],\"props\":{}}\n",
            "{\"from\":\"bob\",\"to\":\"amy\",\"labels\":[\"KNOWS\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let names = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        // Only Amy(25)->Bob(40) satisfies x.age < y.age.
        assert_eq!(
            names("MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age) RETURN x.name AS n"),
            vec!["Amy"]
        );
        // Single-node group with WHERE.
        assert_eq!(
            names("MATCH ((x:Person) WHERE x.age >= 35) RETURN x.name AS n"),
            vec!["Bob"]
        );
        // A named path over an unquantified group is rejected (matches core).
        assert!(
            crate::gql::parse("MATCH p = ((x)-[:KNOWS]->(y) WHERE x.age < y.age) RETURN p")
                .is_err()
        );
    }

    /// The `LET name = expr` clause adds a binding, carrying existing bindings
    /// forward, so a later RETURN/GROUP BY can reference it. t = 5,5,9.
    #[test]
    fn let_clause_binds_and_carries_forward() {
        let mut b = Builder::default();
        b.node(&["P"], &[("t", n(5.0))]);
        b.node(&["P"], &[("t", n(5.0))]);
        b.node(&["P"], &[("t", n(9.0))]);
        let store = b.build();
        let rows = |q: &str| -> Vec<(f64, f64)> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            run(&plan, &store)
                .rows
                .iter()
                .map(|r| match (&r[0], &r[1]) {
                    (Value::Num(a), Value::Num(c)) => (*a, *c),
                    o => panic!("{o:?}"),
                })
                .collect()
        };
        // LET-bound key used in RETURN + GROUP BY + ORDER BY.
        assert_eq!(
            rows("MATCH (n:P) LET t = n.t RETURN t, count(*) AS c GROUP BY t ORDER BY t"),
            vec![(5.0, 2.0), (9.0, 1.0)]
        );
        // The original binding `n` survives the LET (still usable downstream).
        assert_eq!(
            rows("MATCH (n:P) LET t = n.t RETURN n.t AS a, count(*) AS c ORDER BY a"),
            vec![(5.0, 2.0), (9.0, 1.0)]
        );
    }

    /// ORDER BY resolves an output alias even before `NULLS FIRST|LAST`, and ORDER
    /// BY the underlying expression of a projected alias sorts by that output column
    /// (so it composes with DISTINCT). k = 3, (null), 7 over three P nodes.
    #[test]
    fn order_by_alias_with_nulls_and_projected_expr() {
        let mut b = Builder::default();
        b.node(&["P"], &[("k", n(3.0)), ("nn", n(1.0))]);
        b.node(&["P"], &[("nn", n(2.0))]); // k absent -> null
        b.node(&["P"], &[("k", n(7.0)), ("nn", n(1.0))]);
        let store = b.build();
        let col0 = |q: &str| -> Vec<Value> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            run(&plan, &store)
                .rows
                .iter()
                .map(|r| r[0].clone())
                .collect()
        };
        // NULLS FIRST after a bare alias: null sorts first, then 3, 7.
        let got = col0("MATCH (u:P) RETURN u.k AS a ORDER BY a NULLS FIRST");
        assert!(got[0].is_null());
        assert!(matches!(got[1], Value::Num(x) if x == 3.0));
        assert!(matches!(got[2], Value::Num(x) if x == 7.0));
        // DISTINCT with ORDER BY the underlying expression of the projected alias.
        let got = col0("MATCH (u:P) RETURN DISTINCT u.nn AS a ORDER BY u.nn");
        assert_eq!(got.len(), 2); // distinct {1,2}
        assert!(matches!(got[0], Value::Num(x) if x == 1.0));
        assert!(matches!(got[1], Value::Num(x) if x == 2.0));
        // An ORDER BY *expression* may reference an output alias by name (`a` inside
        // a LET-IN): it inlines to the alias's definition (u.k), so the rows sort by
        // k — null first only under NULLS FIRST; default nulls last.
        let got = col0("MATCH (u:P) RETURN u.k AS a ORDER BY (LET x = a IN x END)");
        assert!(matches!(got[0], Value::Num(x) if x == 3.0));
        assert!(matches!(got[1], Value::Num(x) if x == 7.0));
        assert!(got[2].is_null());
    }

    /// An explicit `GROUP BY` after the RETURN list parses and groups the same as
    /// the implicit (non-aggregate items are the keys). n=1,1,2 over three P nodes.
    #[test]
    fn explicit_group_by_after_return() {
        let mut b = Builder::default();
        b.node(&["P"], &[("n", n(1.0))]);
        b.node(&["P"], &[("n", n(1.0))]);
        b.node(&["P"], &[("n", n(2.0))]);
        let store = b.build();
        // GROUP BY the underlying expression, ORDER BY the alias then the expr.
        let rows = |q: &str| -> Vec<(f64, f64)> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            run(&plan, &store)
                .rows
                .iter()
                .map(|r| match (&r[0], &r[1]) {
                    (Value::Num(a), Value::Num(c)) => (*a, *c),
                    o => panic!("{o:?}"),
                })
                .collect()
        };
        assert_eq!(
            rows("MATCH (u:P) RETURN u.n AS a, count(*) AS c GROUP BY u.n ORDER BY a"),
            vec![(1.0, 2.0), (2.0, 1.0)]
        );
        assert_eq!(
            rows("MATCH (u:P) RETURN u.n AS a, count(*) AS c GROUP BY u.n ORDER BY u.n"),
            vec![(1.0, 2.0), (2.0, 1.0)]
        );
    }

    /// GROUP BY a key that is NOT among the RETURN items still groups: it becomes a
    /// hidden grouping key, dropped from the output. `RETURN count(*) GROUP BY
    /// e.dept` yields one row per dept. An aggregate ORDER BY resolves to its true
    /// (keys-then-aggs) schema column, not its RETURN position.
    #[test]
    fn group_by_non_returned_key() {
        let mut b = Builder::default();
        b.node(&["E"], &[("dept", s("eng")), ("sal", n(100.0))]);
        b.node(&["E"], &[("dept", s("eng")), ("sal", n(200.0))]);
        b.node(&["E"], &[("dept", s("sales")), ("sal", n(50.0))]);
        let store = b.build();
        let nums = |q: &str| -> Vec<f64> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            run(&plan, &store)
                .rows
                .iter()
                .map(|r| match &r[0] {
                    Value::Num(x) => *x,
                    o => panic!("{o:?}"),
                })
                .collect()
        };
        // count per dept, key not returned: two groups (eng=2, sales=1), any order.
        let mut c = nums("MATCH (e:E) RETURN count(*) AS c GROUP BY e.dept");
        c.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(c, vec![1.0, 2.0]);
        // sum per dept ordered by the aggregate: the ORDER BY alias must hit the sum
        // column (after the hidden key), so ascending is [50, 300] not [300, 50].
        assert_eq!(
            nums("MATCH (e:E) RETURN sum(e.sal) AS s GROUP BY e.dept ORDER BY s"),
            vec![50.0, 300.0]
        );
    }

    /// A leading `OPTIONAL MATCH` with no prior binding: on an EMPTY graph it still
    /// yields one row, with the pattern variable NULL — so `n.missing IS NULL` is
    /// true. On a non-empty graph it behaves like an ordinary scan (one row per
    /// node), no null padding.
    #[test]
    fn leading_optional_match_pads_one_null_row_when_empty() {
        let empty = Builder::default().build();
        let plan = crate::opt::optimize_indexed(
            crate::gql::parse("OPTIONAL MATCH (n) RETURN n.missing IS NULL AS m").unwrap(),
            &empty,
        );
        let out = run(&plan, &empty);
        assert_eq!(out.rows.len(), 1);
        assert!(matches!(out.rows[0][0], Value::Bool(true)));

        let mut b = Builder::default();
        b.node(&["X"], &[("a", n(5.0))]);
        b.node(&["X"], &[("a", n(6.0))]);
        let store = b.build();
        let plan = crate::opt::optimize_indexed(
            crate::gql::parse("OPTIONAL MATCH (n) RETURN n.a AS a ORDER BY a").unwrap(),
            &store,
        );
        let got: Vec<f64> = run(&plan, &store)
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Num(x) => *x,
                o => panic!("{o:?}"),
            })
            .collect();
        assert_eq!(got, vec![5.0, 6.0]);
    }

    /// LIMIT 0 yields the empty result WITHOUT evaluating the projection, so a
    /// faulting expression (`1/0`) under LIMIT 0 does not error (matches core).
    #[test]
    fn limit_zero_short_circuits_before_projection() {
        let mut b = Builder::default();
        b.node(&["T"], &[("x", n(3.0))]);
        let store = b.build();
        // Without LIMIT 0, `1/0` faults; with it, the projection is never reached.
        let plan = crate::opt::optimize_indexed(
            crate::gql::parse("MATCH (n:T) RETURN 1/0 AS x LIMIT 0").unwrap(),
            &store,
        );
        let out = try_run(&plan, &store).expect("LIMIT 0 must not fault");
        assert_eq!(out.rows.len(), 0);
        // DISTINCT … LIMIT 0 too.
        let plan = crate::opt::optimize_indexed(
            crate::gql::parse("MATCH (n:T) RETURN DISTINCT 1/0 AS x LIMIT 0").unwrap(),
            &store,
        );
        assert_eq!(try_run(&plan, &store).unwrap().rows.len(), 0);
    }

    /// A named path over a NON-shortest var-length pattern binds the walk lineage,
    /// so path_length(p)/edges(p)/nodes(p) resolve. Fixture a->b->c->a (cycle) + a->d.
    #[test]
    fn named_path_over_var_length_binds_lineage() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"c\",\"to\":\"a\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"a\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let run_q = |q: &str, col: usize| -> Vec<f64> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v: Vec<f64> = run(&plan, &store)
                .rows
                .iter()
                .map(|r| match r[col] {
                    Value::Num(x) => x,
                    ref o => panic!("{o:?}"),
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        // paths from a of length 1..3: a-b (1), a-d (1), a-b-c (2), a-b-c-a (3).
        assert_eq!(
            run_q(
                "MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN path_length(p) AS len",
                0
            ),
            vec![1.0, 1.0, 2.0, 3.0]
        );
        // size(edges(p)) tracks the hop count (path_length).
        assert_eq!(
            run_q(
                "MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN size(edges(p)) AS es",
                0
            ),
            vec![1.0, 1.0, 2.0, 3.0]
        );
        // min 0 binds the length-0 seed path (a itself) too.
        assert_eq!(
            run_q(
                "MATCH p = (a:N {id:'a'})-[:R]->{0,1}(x) RETURN path_length(p) AS len",
                0
            ),
            vec![0.0, 1.0, 1.0]
        );
    }

    /// String (K10) and list/element (K11) functions match hand-computed values.
    #[test]
    fn added_string_and_list_functions() {
        let mut b = Builder::default();
        b.node(&["N", "M"], &[("z", n(1.0)), ("a", n(2.0))]);
        let store = b.build();
        let val = |e: &str| -> Value {
            let q = format!("MATCH (x:N) RETURN {e} AS v");
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
        };
        let str_of = |e: &str| match val(e) {
            Value::Str(s) => s.to_string(),
            o => panic!("{e} → {o:?}"),
        };
        // `Value` has no `PartialEq` (the value contract owns equality), so compare
        // list contents via debug strings.
        let list_of = |e: &str| -> Vec<String> {
            match val(e) {
                Value::List(v) => v.iter().map(|x| format!("{x:?}")).collect(),
                o => panic!("{e} → {o:?}"),
            }
        };
        let dbg = |xs: &[Value]| -> Vec<String> { xs.iter().map(|x| format!("{x:?}")).collect() };
        // trims (whitespace, and explicit char set)
        assert_eq!(str_of("ltrim('  hi ')"), "hi ");
        assert_eq!(str_of("rtrim('  hi ')"), "  hi");
        assert_eq!(str_of("btrim('xxhixx', 'x')"), "hi");
        // reverse (string + list), left/right, split
        assert_eq!(str_of("reverse('abc')"), "cba");
        assert_eq!(str_of("left('abcd', 2)"), "ab");
        assert_eq!(str_of("right('abcd', 2)"), "cd");
        assert_eq!(str_of("left('ab', 5)"), "ab"); // n > len → whole
        assert_eq!(
            list_of("split('a,b,c', ',')"),
            dbg(&[s("a"), s("b"), s("c")])
        );
        // list fns
        assert_eq!(
            list_of("reverse([1, 2, 3])"),
            dbg(&[n(3.0), n(2.0), n(1.0)])
        );
        assert_eq!(list_of("tail([1, 2, 3])"), dbg(&[n(2.0), n(3.0)]));
        assert_eq!(
            list_of("range(1, 4)"),
            dbg(&[n(1.0), n(2.0), n(3.0), n(4.0)])
        );
        assert_eq!(list_of("range(5, 1, -1)").len(), 5);
        assert!(val("range(1, 4, 0)").is_null()); // zero step
        assert_eq!(list_of("range(5, 1)"), Vec::<String>::new()); // wrong-sign default step
                                                                  // element fns: keys (sorted present props), labels (sorted)
        assert_eq!(list_of("keys(x)"), dbg(&[s("a"), s("z")]));
        assert_eq!(list_of("labels(x)"), dbg(&[s("M"), s("N")]));
    }

    /// `IN` / `NOT IN` over a list literal (K7), desugared to an OR-chain of
    /// equals — including three-valued behavior with a NULL in the list.
    #[test]
    fn in_operator() {
        let mut b = Builder::default();
        b.node(&["N"], &[("a", n(1.0))]);
        b.node(&["N"], &[("a", n(2.0))]);
        b.node(&["N"], &[("a", n(9.0))]);
        b.node(&["N"], &[]); // a is NULL
        let store = b.build();
        let ids = |q: &str| -> Vec<String> {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &store), 0);
            v.sort();
            v
        };
        // a IN [1,2] → the 1 and 2 nodes.
        assert_eq!(
            ids("MATCH (n:N) WHERE n.a IN [1, 2] RETURN n.a AS a"),
            vec!["Num(1.0)", "Num(2.0)"]
        );
        // NOT IN → the 9 node only (NULL-a is UNKNOWN, dropped, not returned).
        assert_eq!(
            ids("MATCH (n:N) WHERE n.a NOT IN [1, 2] RETURN n.a AS a"),
            vec!["Num(9.0)"]
        );
        // A NULL element makes a non-match UNKNOWN → row drops (3VL): only the
        // literal 1 matches; 2/9 are UNKNOWN (could equal the null), dropped.
        assert_eq!(
            ids("MATCH (n:N) WHERE n.a IN [1, null] RETURN n.a AS a"),
            vec!["Num(1.0)"]
        );
        // Empty list → nobody matches.
        assert_eq!(
            ids("MATCH (n:N) WHERE n.a IN [] RETURN n.a AS a"),
            Vec::<String>::new()
        );
    }

    /// Dynamic (non-literal) IN over a list PROPERTY — the runtime `Expr::In`, with
    /// the same three-valued behavior as the literal OR-chain.
    #[test]
    fn in_operator_dynamic() {
        let mut b = Builder::default();
        b.node(
            &["N"],
            &[
                ("a", n(2.0)),
                ("xs", Value::List(vec![n(1.0), n(2.0), n(3.0)])),
            ],
        );
        b.node(
            &["N"],
            &[
                ("a", n(9.0)),
                ("xs", Value::List(vec![n(1.0), n(2.0), n(3.0)])),
            ],
        );
        b.node(
            &["N"],
            &[
                ("a", n(5.0)),
                ("xs", Value::List(vec![n(1.0), Value::Null, n(3.0)])),
            ],
        );
        let store = b.build();
        let ids = |q: &str| -> Vec<String> {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &store), 0);
            v.sort();
            v
        };
        // n.a IN n.xs: only the a=2 node (2 ∈ [1,2,3]); a=9 not in; a=5 vs [1,null,3]
        // is UNKNOWN (null element) → dropped.
        assert_eq!(
            ids("MATCH (n:N) WHERE n.a IN n.xs RETURN n.a AS a"),
            vec!["Num(2.0)"]
        );
        // 2 IN n.xs: the two nodes whose list has 2; the [1,null,3] node lacks 2 and
        // is UNKNOWN → dropped.
        assert_eq!(
            ids("MATCH (n:N) WHERE 2 IN n.xs RETURN n.a AS a"),
            vec!["Num(2.0)", "Num(9.0)"]
        );
    }

    /// Undirected `~` traversal is `Dir::Both`: a normal edge is reached from both
    /// endpoints (two rows), but a self-loop is walked ONCE (its in-side copy is
    /// dropped), matching core's `SelfLoops::Once`.
    #[test]
    fn undirected_tilde_self_loop_counted_once() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"a\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e2\",\"from\":\"a\",\"to\":\"b\",\"type\":\"R\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let count = |q: &str| -> f64 {
            match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(n) => n,
                ref other => panic!("want num, got {other:?}"),
            }
        };
        // Self-loop once (1) + a-b both orientations (2) = 3.
        assert_eq!(count("MATCH (a)~[r]~(b) RETURN count(*) AS c"), 3.0);
        // The same over a single-hop var-length spelling routes through the DFS
        // walker, which also drops the self-loop's in-side copy.
        assert_eq!(count("MATCH (a)~[:R]~{1,1}(b) RETURN count(*) AS c"), 3.0);
        // A directed self-loop is walked once either way (one index touched).
        assert_eq!(count("MATCH (a)-[r:R]->(b) RETURN count(*) AS c"), 2.0);
    }

    /// `SELECT … GROUP BY … HAVING …` filters grouped rows: on an aggregate
    /// (`count(*) > 1`), on a group key (`n.age >= 35`), globally (no GROUP BY), and
    /// `HAVING null` drops every group. An aggregate may appear only in HAVING.
    #[test]
    fn select_having() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"Person\"],\"props\":{\"age\":30}}\n",
            "{\"id\":\"b\",\"labels\":[\"Person\"],\"props\":{\"age\":30}}\n",
            "{\"id\":\"c\",\"labels\":[\"Person\"],\"props\":{\"age\":40}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let rows = |q: &str| -> Vec<String> {
            run(&crate::gql::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| {
                    r.iter()
                        .map(|c| format!("{c:?}"))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .collect()
        };
        // HAVING on an aggregate: only the age-30 group (count 2 > 1).
        assert_eq!(
            rows("SELECT n.age AS age, count(*) AS c FROM MATCH (n:Person) GROUP BY n.age HAVING count(*) > 1 ORDER BY age"),
            vec!["Num(30.0),Num(2.0)"]
        );
        // Aggregate only in HAVING, not in the SELECT list.
        assert_eq!(
            rows("SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age HAVING count(*) > 1"),
            vec!["Num(30.0)"]
        );
        // HAVING on a group key.
        assert_eq!(
            rows("SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age HAVING n.age >= 35 ORDER BY age"),
            vec!["Num(40.0)"]
        );
        // Global HAVING (no GROUP BY): 3 people — passes >2, fails >100.
        assert_eq!(
            rows("SELECT count(*) AS c FROM MATCH (n:Person) HAVING count(*) > 2"),
            vec!["Num(3.0)"]
        );
        assert!(
            rows("SELECT count(*) AS c FROM MATCH (n:Person) HAVING count(*) > 100").is_empty()
        );
        // HAVING null drops every group.
        assert!(
            rows("SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age HAVING null").is_empty()
        );
    }

    /// `ALL SHORTEST` emits one row per distinct shortest path (so a target reached
    /// by two equal-length paths appears twice), while `ANY SHORTEST` emits one row
    /// per reachable target. A `*` quantifier includes the zero-length seed.
    #[test]
    fn all_shortest_multiplicity() {
        // Diamond: a->b, a->c, b->d, c->d — d is reachable by two 2-hop paths.
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
            "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"b\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e2\",\"from\":\"a\",\"to\":\"c\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e3\",\"from\":\"b\",\"to\":\"d\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e4\",\"from\":\"c\",\"to\":\"d\",\"type\":\"R\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let count = |q: &str| -> f64 {
            match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(n) => n,
                ref other => panic!("want num, got {other:?}"),
            }
        };
        // ANY: seed a (len 0) + b + c + d(once) = 4 rows.
        assert_eq!(
            count("MATCH ANY SHORTEST (a {id:'a'})-[:R]->*(x) RETURN count(*) AS c"),
            4.0
        );
        // ALL: a + b + c + d TWICE (two shortest paths) = 5 rows.
        assert_eq!(
            count("MATCH ALL SHORTEST (a {id:'a'})-[:R]->*(x) RETURN count(*) AS c"),
            5.0
        );
        // ALL restricted to endpoint d: two shortest paths → 2 rows.
        assert_eq!(
            count("MATCH ALL SHORTEST (a {id:'a'})-[:R]->*(x {id:'d'}) RETURN count(*) AS c"),
            2.0
        );
        // SHORTEST 1 reduces to ANY (one row for d); SHORTEST 1 GROUP to ALL (two).
        assert_eq!(
            count("MATCH SHORTEST 1 (a {id:'a'})-[:R]->*(x {id:'d'}) RETURN count(*) AS c"),
            1.0
        );
        assert_eq!(
            count("MATCH SHORTEST 1 GROUP (a {id:'a'})-[:R]->*(x {id:'d'}) RETURN count(*) AS c"),
            2.0
        );
    }

    /// `SELECT … [FROM MATCH …]` is sugar for MATCH…RETURN: a constant projection
    /// with no FROM, a plain projection, a global aggregate with WHERE, and a
    /// GROUP BY (via implicit grouping) with ORDER BY over an output alias.
    #[test]
    fn select_from_match() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Alice\",\"age\":30}}\n",
            "{\"id\":\"b\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Bob\",\"age\":40}}\n",
            "{\"id\":\"c\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Cara\",\"age\":30}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let one =
            |q: &str| -> Value { run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() };
        // Constant projection, no FROM.
        assert!(matches!(one("SELECT 1 + 2 AS v"), Value::Num(n) if n == 3.0));
        // Plain projection with an inline filter.
        assert!(
            matches!(one("SELECT n.name AS nm FROM MATCH (n:Person {name: 'Alice'})"), Value::Str(s) if &*s == "Alice")
        );
        // Global aggregate with WHERE (>= 30 → all three).
        assert!(
            matches!(one("SELECT count(*) AS c FROM MATCH (n:Person) WHERE n.age >= 30"), Value::Num(n) if n == 3.0)
        );
        // GROUP BY age with ORDER BY the output alias: ages 30 (×2), 40 (×1).
        let grouped = run(
            &crate::gql::parse(
                "SELECT n.age AS age, count(*) AS c FROM MATCH (n:Person) GROUP BY n.age ORDER BY age",
            )
            .unwrap(),
            &store,
        );
        let rows: Vec<String> = grouped
            .rows
            .iter()
            .map(|r| format!("{:?},{:?}", r[0], r[1]))
            .collect();
        assert_eq!(rows, vec!["Num(30.0),Num(2.0)", "Num(40.0),Num(1.0)"]);
    }

    /// A NaN operand makes ordering (`< > <= >=`) definitely FALSE (IEEE), NOT unknown —
    /// matching JS and the pure-TS engine. Equality with NaN stays false (NaN != NaN).
    #[test]
    fn nan_ordering_is_ieee_false() {
        let store =
            crate::ndjson::from_ndjson("{\"id\":\"n\",\"labels\":[\"V\"],\"props\":{}}").unwrap();
        let val =
            |q: &str| -> Value { run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() };
        // log10(-1) is NaN. Every ordering against it is FALSE, not null.
        assert!(matches!(
            val("RETURN (log10(-1) < 5) AS x"),
            Value::Bool(false)
        ));
        assert!(matches!(
            val("RETURN (log10(-1) >= 5) AS x"),
            Value::Bool(false)
        ));
        assert!(matches!(
            val("RETURN (5 < log10(-1)) AS x"),
            Value::Bool(false)
        ));
        assert!(matches!(
            val("RETURN (0.0 > log10(-1)) AS x"),
            Value::Bool(false)
        ));
        // Equality with NaN is still FALSE, and its negation TRUE — ordering is the only
        // thing that changed from 3-valued to IEEE.
        assert!(matches!(
            val("RETURN (log10(-1) = log10(-1)) AS x"),
            Value::Bool(false)
        ));
        assert!(matches!(
            val("RETURN (log10(-1) <> log10(-1)) AS x"),
            Value::Bool(true)
        ));
    }

    /// -0 and +0 are one value: atan2 (the only fn whose result distinguishes the sign of
    /// a zero operand) folds -0 to +0 on both inputs, whether the -0 came from a literal
    /// or from arithmetic (`0 * -1`). So atan2(±0, -1) is always +PI, never -PI.
    #[test]
    fn signed_zero_folds_in_atan2() {
        let store =
            crate::ndjson::from_ndjson("{\"id\":\"n\",\"labels\":[\"V\"],\"props\":{}}").unwrap();
        let num = |q: &str| -> f64 {
            match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() {
                Value::Num(n) => n,
                other => panic!("want num, got {other:?}"),
            }
        };
        let pi = std::f64::consts::PI;
        assert!((num("RETURN atan2(-0.0, -1.0) AS r") - pi).abs() < 1e-12);
        assert!((num("RETURN atan2(0.0 * -1, -1.0) AS r") - pi).abs() < 1e-12);
        assert!(num("RETURN atan2(-0.0, -0.0) AS r").abs() < 1e-12); // atan2(+0, +0) = 0
    }

    /// NaN has no sign: `sign(NaN)` is NaN (→ null at egress), NOT the 0 that a naive
    /// `>0 / <0 / else` would give. Finite inputs are unchanged.
    #[test]
    fn sign_of_nan_is_nan() {
        let store =
            crate::ndjson::from_ndjson("{\"id\":\"n\",\"labels\":[\"V\"],\"props\":{}}").unwrap();
        let val =
            |q: &str| -> Value { run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() };
        // asin(1e100) and log10(-1) are NaN; sign keeps NaN at the row level (it is only
        // coerced to null at the JSON egress boundary, per the K4 policy).
        assert!(matches!(val("RETURN sign(asin(1e100)) AS x"), Value::Num(n) if n.is_nan()));
        assert!(matches!(val("RETURN sign(log10(-1)) AS x"), Value::Num(n) if n.is_nan()));
        // Finite inputs are unaffected.
        assert!(matches!(val("RETURN sign(-5.0) AS x"), Value::Num(n) if n == -1.0));
        assert!(matches!(val("RETURN sign(5.0) AS x"), Value::Num(n) if n == 1.0));
        assert!(matches!(val("RETURN sign(0.0) AS x"), Value::Num(n) if n == 0.0));
    }

    /// Scalar functions: 2-arg round (incl. negative digits), atan2 (arg order +
    /// null propagation), log10, TRIM spec forms, and list_sort with order/nullOrder.
    #[test]
    fn scalar_fns_batch() {
        let store =
            crate::ndjson::from_ndjson("{\"id\":\"n\",\"labels\":[\"V\"],\"props\":{}}").unwrap();
        let val =
            |q: &str| -> Value { run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() };
        let num = |q: &str| -> f64 {
            match val(q) {
                Value::Num(n) => n,
                other => panic!("want num, got {other:?}"),
            }
        };
        // round to N decimal places; negative digits round left of the point.
        assert_eq!(num("RETURN round(1.2345, 2) AS r"), 1.23);
        assert_eq!(num("RETURN round(1234.5678, -2) AS r"), 1200.0);
        assert_eq!(num("RETURN round(2.5) AS r"), 3.0); // 1-arg still works
                                                        // atan2(y, x): arg order matters; a null arg → NULL.
        assert!((num("RETURN atan2(1, 1) AS r") - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert_eq!(num("RETURN atan2(0, 1) AS r"), 0.0);
        assert!(matches!(val("RETURN atan2(null, 1) AS r"), Value::Null));
        // log10.
        assert_eq!(num("RETURN log10(1000) AS r"), 3.0);
        // TRIM spec forms desugar to trim/ltrim/rtrim with the char as 2nd arg.
        let s = |q: &str| -> String {
            match val(q) {
                Value::Str(x) => x.to_string(),
                other => panic!("want str, got {other:?}"),
            }
        };
        assert_eq!(s("RETURN TRIM('  hi  ') AS r"), "hi");
        assert_eq!(s("RETURN TRIM(BOTH FROM '  hi  ') AS r"), "hi");
        assert_eq!(s("RETURN TRIM(LEADING 'x' FROM 'xxhi') AS r"), "hi");
        assert_eq!(s("RETURN TRIM(TRAILING 'x' FROM 'hixx') AS r"), "hi");
        assert_eq!(s("RETURN TRIM('x' FROM 'xxhixx') AS r"), "hi");
        // list_sort: default ascending, 'desc' reverses, nullOrder places nulls.
        // Compare list results by their debug rendering (Value is not PartialEq).
        let list = |q: &str| -> String { format!("{:?}", val(q)) };
        assert_eq!(
            list("RETURN list_sort([3,1,2], 'desc') AS r"),
            "List([Num(3.0), Num(2.0), Num(1.0)])"
        );
        assert_eq!(
            list("RETURN list_sort([3,1,null,2], 'asc', 'first') AS r"),
            "List([Null, Num(1.0), Num(2.0), Num(3.0)])"
        );
        // default null placement is LAST.
        assert_eq!(
            list("RETURN list_sort([2,null,1]) AS r"),
            "List([Num(1.0), Num(2.0), Null])"
        );
    }

    /// An edge-type disjunction `-[:A|B]->` matches an edge whose type is ANY of the
    /// listed types; a typed-but-all-unknown disjunction matches nothing (it is NOT
    /// read as "any"); an unknown name in a partial disjunction is dropped.
    #[test]
    fn edge_type_disjunction() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"b\",\"type\":\"KNOWS\",\"props\":{}}\n",
            "{\"id\":\"e2\",\"from\":\"a\",\"to\":\"c\",\"type\":\"CREATED\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let count = |q: &str| -> f64 {
            match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(n) => n,
                ref other => panic!("want num, got {other:?}"),
            }
        };
        // Both edge types match → both neighbours.
        assert_eq!(
            count("MATCH (a)-[:KNOWS|CREATED]->(x) RETURN count(*) AS c"),
            2.0
        );
        // Order is irrelevant to the set.
        assert_eq!(
            count("MATCH (a)-[:CREATED|KNOWS]->(x) RETURN count(*) AS c"),
            2.0
        );
        // A single named type still matches only that one.
        assert_eq!(count("MATCH (a)-[:KNOWS]->(x) RETURN count(*) AS c"), 1.0);
        // A partial disjunction drops the unknown name, keeping the known one.
        assert_eq!(
            count("MATCH (a)-[:KNOWS|BOGUS]->(x) RETURN count(*) AS c"),
            1.0
        );
        // Typed but ALL-unknown matches nothing (NOT read as "any type").
        assert_eq!(
            count("MATCH (a)-[:BOGUS|NOPE]->(x) RETURN count(*) AS c"),
            0.0
        );
    }

    /// `MATCH WALK` lets a variable-length hop reuse an edge; `TRAIL` (the default)
    /// forbids it. Over a self-loop, a length-2 hop exists as a WALK (reuse the loop)
    /// but not as a TRAIL.
    #[test]
    fn path_mode_walk_vs_trail_edge_reuse() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"a\",\"type\":\"R\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let count = |q: &str| -> f64 {
            match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(n) => n,
                ref other => panic!("want num, got {other:?}"),
            }
        };
        // WALK: a->a->a reuses the loop edge — one length-2 walk.
        assert_eq!(
            count("MATCH WALK (a {id:'a'})-[:R]->{2,2}(x) RETURN count(*) AS c"),
            1.0
        );
        // TRAIL (default): the loop edge can't repeat — no length-2 trail.
        assert_eq!(
            count("MATCH TRAIL (a {id:'a'})-[:R]->{2,2}(x) RETURN count(*) AS c"),
            0.0
        );
        assert_eq!(
            count("MATCH (a {id:'a'})-[:R]->{2,2}(x) RETURN count(*) AS c"),
            0.0
        );
    }

    /// `~` resolves to `Dir::Both` regardless of which side (or a `-`/`~` mix) is
    /// used, matching either traversal direction of the edge.
    #[test]
    fn undirected_tilde_matches_either_direction() {
        let nd = concat!(
            "{\"id\":\"josh\",\"labels\":[\"P\"],\"props\":{\"name\":\"josh\"}}\n",
            "{\"id\":\"vadas\",\"labels\":[\"P\"],\"props\":{\"name\":\"vadas\"}}\n",
            "{\"id\":\"e1\",\"from\":\"josh\",\"to\":\"vadas\",\"type\":\"KNOWS\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        // josh has an OUT edge; the undirected walk still reaches vadas.
        let mut a = names_of(
            &run(
                &crate::gql::parse(
                    "MATCH (a)~[:KNOWS]~(b) WHERE a.name = 'josh' RETURN b.name AS n",
                )
                .unwrap(),
                &store,
            ),
            0,
        );
        a.sort();
        assert_eq!(a, vec!["vadas"]);
        // vadas has only an IN edge; the undirected walk reaches josh.
        let b = names_of(
            &run(
                &crate::gql::parse(
                    "MATCH (a)~[:KNOWS]~(b) WHERE a.name = 'vadas' RETURN b.name AS n",
                )
                .unwrap(),
                &store,
            ),
            0,
        );
        assert_eq!(b, vec!["josh"]);
    }

    /// External ids are PRESERVED through ingest and returned by element_id (nodes
    /// and edges), and survive an NDJSON round-trip.
    #[test]
    fn element_id_preserves_external_ids() {
        let nd = concat!(
            "{\"id\":\"alice\",\"labels\":[\"P\"],\"props\":{}}\n",
            "{\"id\":\"bob\",\"labels\":[\"P\"],\"props\":{}}\n",
            "{\"id\":\"e42\",\"from\":\"alice\",\"to\":\"bob\",\"type\":\"KNOWS\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        // element_id(node) returns the preserved string id.
        let mut ns = names_of(
            &run(
                &crate::gql::parse("MATCH (n:P) RETURN element_id(n) AS a0").unwrap(),
                &store,
            ),
            0,
        );
        ns.sort();
        assert_eq!(ns, vec!["alice", "bob"]);
        // element_id(edge) returns the preserved edge id.
        let es = run(
            &crate::gql::parse("MATCH (a:P)-[r:KNOWS]->(b) RETURN element_id(r) AS a0").unwrap(),
            &store,
        );
        assert!(matches!(&es.rows[0][0], Value::Str(s) if &**s == "e42"));
        // NDJSON round-trip preserves those ids (dump contains them, reload keeps).
        let dump = crate::ndjson::to_ndjson(&store);
        assert!(dump.contains("\"id\":\"alice\"") && dump.contains("\"id\":\"e42\""));
        assert_eq!(
            crate::ndjson::to_ndjson(&crate::ndjson::from_ndjson(&dump).unwrap()),
            dump
        );
    }

    /// `type(edge)` and the list-algebra functions (previously deferred) match
    /// hand-computed values.
    #[test]
    fn type_and_list_algebra_functions() {
        let mut b = Builder::default();
        let x = b.node(&["N"], &[]);
        let y = b.node(&["N"], &[]);
        b.edge(x, y, "KNOWS");
        let store = b.build();
        // type(edge)
        let t = run(
            &crate::gql::parse("MATCH (a:N)-[r:KNOWS]->(b) RETURN type(r) AS t").unwrap(),
            &store,
        );
        assert!(matches!(&t.rows[0][0], Value::Str(s) if &**s == "KNOWS"));

        let list = |e: &str| -> Vec<String> {
            let q = format!("MATCH (a:N) RETURN {e} AS v LIMIT 1");
            match &run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0] {
                Value::List(v) => v.iter().map(|x| format!("{x:?}")).collect(),
                o => panic!("{e} → {o:?}"),
            }
        };
        let one = |e: &str| -> Value {
            let q = format!("MATCH (a:N) RETURN {e} AS v LIMIT 1");
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
        };
        let dbg = |xs: &[Value]| -> Vec<String> { xs.iter().map(|x| format!("{x:?}")).collect() };
        assert_eq!(list("append([1, 2], 3)"), dbg(&[n(1.0), n(2.0), n(3.0)]));
        assert!(matches!(one("list_contains([1, 2, 3], 2)"), Value::Num(x) if x == 1.0));
        assert!(matches!(one("list_contains([1, 2], 5)"), Value::Num(x) if x == 0.0));
        assert!(matches!(one("list_contains([1, null], null)"), Value::Num(x) if x == 1.0));
        assert_eq!(list("list_sort([3, 1, 2])"), dbg(&[n(1.0), n(2.0), n(3.0)]));
        assert_eq!(
            list("list_union([1, 1, 2], [2, 3])"),
            dbg(&[n(1.0), n(2.0), n(3.0)])
        );
        assert_eq!(
            list("difference([1, 1, 2, 3], [2])"),
            dbg(&[n(1.0), n(3.0)])
        );
        assert_eq!(
            list("intersection([1, 2, 2, 3], [2, 3, 4])"),
            dbg(&[n(2.0), n(3.0)])
        );
    }

    /// Late materialization (sorted top-K over a Project) returns the SAME rows as
    /// the eager path — the non-key column is projected only for the survivors.
    #[test]
    fn late_materialize_top_k_matches_eager() {
        let mut b = Builder::default();
        for i in 0..50u32 {
            b.node(
                &["P"],
                &[("age", n(f64::from(i % 10))), ("name", s(&format!("p{i}")))],
            );
        }
        let store = b.build();
        // Top-3 by age DESC, then name — a non-key column (name) is projected.
        let q = "MATCH (p:P) RETURN p.name AS name, p.age AS age ORDER BY age DESC, name LIMIT 3";
        let got = run(&crate::gql::parse(q).unwrap(), &store);
        // Highest ages are 9 (p9, p19, p29, p39, p49) → name-sorted first three.
        assert_eq!(names_of(&got, 0), vec!["p19", "p29", "p39"]);
        assert!(got
            .rows
            .iter()
            .all(|r| matches!(r[1], Value::Num(x) if x == 9.0)));
        // With SKIP: rows 3..6 of the same order.
        let q2 = "MATCH (p:P) RETURN p.name AS name, p.age AS age ORDER BY age DESC, name SKIP 3 LIMIT 2";
        assert_eq!(
            names_of(&run(&crate::gql::parse(q2).unwrap(), &store), 0),
            vec!["p49", "p9"]
        );
    }

    /// A low-cardinality string column dictionary-encodes, and every read shape
    /// (DISTINCT / GROUP BY / equality filter / ORDER BY) returns exactly what the
    /// plain `Str` column would — while a high-cardinality column stays `Str`.
    #[test]
    fn dict_encoded_column_round_trips() {
        let depts = ["eng", "sales", "ops"];
        let mut b = Builder::default();
        for i in 0..30u32 {
            b.node(
                &["P"],
                &[
                    ("dept", s(depts[i as usize % 3])),
                    ("name", s(&format!("p{i}"))), // 30 distinct -> stays Str
                ],
            );
        }
        let store = b.build();
        // The low-card column encoded; the high-card one did not.
        assert!(matches!(
            store.column("dept"),
            Some(crate::store::Column::Dict { .. })
        ));
        assert!(matches!(
            store.column("name"),
            Some(crate::store::Column::Str { .. })
        ));

        let rows = |q: &str| {
            let mut r: Vec<String> = names_of(&run(&crate::gql::parse(q).unwrap(), &store), 0);
            r.sort();
            r
        };
        // DISTINCT over the dict column.
        assert_eq!(
            rows("MATCH (n:P) RETURN DISTINCT n.dept AS d"),
            vec!["eng", "ops", "sales"]
        );
        // GROUP BY the dict column: 10 of each.
        let g = run(
            &crate::gql::parse("MATCH (n:P) RETURN n.dept AS d, count(*) AS c").unwrap(),
            &store,
        );
        assert_eq!(g.rows.len(), 3);
        assert!(g
            .rows
            .iter()
            .all(|r| matches!(r[1], Value::Num(x) if x == 10.0)));
        // count(DISTINCT) over the dict column.
        let c = run(
            &crate::gql::parse("MATCH (n:P) RETURN count(DISTINCT n.dept) AS c").unwrap(),
            &store,
        );
        assert!(matches!(c.rows[0][0], Value::Num(x) if x == 3.0));
        // Equality filter resolves through the dict; a miss matches nothing.
        let count_where = |q: &str| match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(x) => x,
            _ => panic!("count is not a number"),
        };
        assert_eq!(
            count_where("MATCH (n:P) WHERE n.dept = 'eng' RETURN count(*) AS c"),
            10.0
        );
        assert_eq!(
            count_where("MATCH (n:P) WHERE n.dept = 'zzz' RETURN count(*) AS c"),
            0.0
        );
        // ORDER BY the dict column sorts by VALUE, not code.
        let o = run(
            &crate::gql::parse("MATCH (n:P) RETURN DISTINCT n.dept AS d ORDER BY d").unwrap(),
            &store,
        );
        assert_eq!(names_of(&o, 0), vec!["eng", "ops", "sales"]);
    }

    /// Writing a value to a dict-encoded column decodes it to `Str` in place, and the
    /// new value reads back correctly alongside the untouched ones.
    #[test]
    fn dict_column_decodes_on_write() {
        let mut b = Builder::default();
        for _ in 0..6u32 {
            b.node(&["P"], &[("dept", s("eng"))]);
        }
        let mut store = b.build();
        assert!(matches!(
            store.column("dept"),
            Some(crate::store::Column::Dict { .. })
        ));
        let id = store.nodes_with_label("P")[0];
        store.set_prop(id, "dept", s("legal"));
        assert!(matches!(
            store.column("dept"),
            Some(crate::store::Column::Str { .. })
        ));
        assert!(matches!(store.prop(id, "dept"), Value::Str(x) if &*x == "legal"));
        let other = store.nodes_with_label("P")[1];
        assert!(matches!(store.prop(other, "dept"), Value::Str(x) if &*x == "eng"));
    }

    /// Multi-column `DISTINCT` over a dict-encoded string column plus a numeric one
    /// (and an absent cell) dedups on the composite code+bits key exactly as the
    /// general byte-key path would — same distinct tuples, absence as its own value.
    #[test]
    fn multi_col_distinct_over_dict_and_num() {
        let depts = ["eng", "sales"];
        let mut b = Builder::default();
        // 20 rows: dept in {eng,sales} (cycles every row), age in {30,40} (flips
        // every 2 rows) — decoupled, so all 4 present tuples occur...
        for i in 0..20u32 {
            b.node(
                &["P"],
                &[
                    ("dept", s(depts[i as usize % 2])),
                    ("age", n(f64::from(30 + ((i / 2) % 2) * 10))),
                ],
            );
        }
        // ...plus two rows whose dept is ABSENT (age 30) -> a 5th tuple (Null, 30).
        b.node(&["P"], &[("age", n(30.0))]);
        b.node(&["P"], &[("age", n(30.0))]);
        let store = b.build();
        assert!(matches!(
            store.column("dept"),
            Some(crate::store::Column::Dict { .. })
        ));

        let out = run(
            &crate::gql::parse("MATCH (n:P) RETURN DISTINCT n.dept AS d, n.age AS age").unwrap(),
            &store,
        );
        // Render each (dept, age) tuple to a stable string and compare as a set.
        let mut got: Vec<String> = out
            .rows
            .iter()
            .map(|r| format!("{:?}|{:?}", r[0], r[1]))
            .collect();
        got.sort();
        let mut want = vec![
            format!("{:?}|{:?}", Value::Str("eng".into()), Value::Num(30.0)),
            format!("{:?}|{:?}", Value::Str("eng".into()), Value::Num(40.0)),
            format!("{:?}|{:?}", Value::Str("sales".into()), Value::Num(30.0)),
            format!("{:?}|{:?}", Value::Str("sales".into()), Value::Num(40.0)),
            format!("{:?}|{:?}", Value::Null, Value::Num(30.0)),
        ];
        want.sort();
        assert_eq!(got, want);
    }

    // --- relational core (unchanged behavior, now slot-addressed) ---

    #[test]
    fn scan_label_and_project() {
        let store = social();
        let out = run(
            &scan("Person").project(vec![("name".into(), prop(0, "name"))]),
            &store,
        );
        assert_eq!(out.rows.len(), 3);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "bob", "carol"]);
    }

    #[test]
    fn cast_projects_per_row() {
        let store = social();
        // Cast each Person's numeric age to INTEGER (identity here; the ages are
        // already whole) — verifies the per-row Cast arm wires through Project.
        let plan = scan("Person").project(vec![(
            "a".into(),
            Expr::Cast {
                target: crate::ir::CastTarget::Integer,
                expr: Box::new(prop(0, "age")),
            },
        )]);
        let out = run(&plan, &store);
        let mut got: Vec<f64> = out
            .rows
            .iter()
            .map(|r| match r[0] {
                Value::Num(x) => x,
                ref o => panic!("expected Num, got {o:?}"),
            })
            .collect();
        got.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        assert_eq!(got, vec![25.0, 30.0, 40.0]);
    }

    #[test]
    fn cast_fault_surfaces_through_try_run() {
        let store = social();
        // "alice" has no numeric form → the CAST throws E_INVALID_VALUE, and the
        // fallible `try_run` returns that Err (this is why the read pipeline
        // threads Result at all; `run` would panic on the same plan).
        let plan = scan("Person").project(vec![(
            "n".into(),
            Expr::Cast {
                target: crate::ir::CastTarget::Integer,
                expr: Box::new(prop(0, "name")),
            },
        )]);
        let err = try_run(&plan, &store).unwrap_err();
        assert!(err.contains("E_INVALID_VALUE"), "got: {err}");
    }

    #[test]
    fn is_null_projects_definite_bools() {
        // A scan of all nodes: the three Persons carry `age`, the Project node
        // does not. `age IS NULL` must be a definite Bool for EVERY row (never a
        // Null/UNKNOWN), TRUE only where the value is absent.
        let store = social();
        let plan = Plan::Scan { label: None }.project(vec![(
            "n".into(),
            Expr::IsNull {
                expr: Box::new(prop(0, "age")),
                negated: false,
            },
        )]);
        let out = run(&plan, &store);
        // Every value is a concrete boolean — none is Null.
        assert!(out.rows.iter().all(|r| matches!(r[0], Value::Bool(_))));
        let trues = out
            .rows
            .iter()
            .filter(|r| matches!(r[0], Value::Bool(true)))
            .count();
        assert_eq!(trues, 1); // only the Project node lacks `age`
    }

    #[test]
    fn property_exists_separates_present_null_from_absent() {
        // node 0: age present-null, node 1: age absent. PROPERTY_EXISTS is a
        // presence test, so it is TRUE for the present-null and FALSE for absent —
        // the distinction `IS NOT NULL` (both FALSE) cannot draw.
        let mut b = Builder::default();
        b.node(&["P"], &[("name", s("null"))]);
        b.node(&["P"], &[("name", s("absent"))]);
        let mut store = b.build();
        store.set_prop(0, "age", Value::Null);

        let exists = Plan::Scan {
            label: Some("P".into()),
        }
        .project(vec![(
            "e".into(),
            Expr::PropertyExists {
                slot: 0,
                key: "age".into(),
            },
        )]);
        let out = run(&exists, &store);
        assert!(matches!(out.rows[0][0], Value::Bool(true))); // present-null
        assert!(matches!(out.rows[1][0], Value::Bool(false))); // absent
    }

    /// PROPERTY_EXISTS works on an EDGE slot (not just nodes), and a NULL element
    /// (the OPTIONAL unmatched sentinel) yields NULL, not FALSE — matching core.
    #[test]
    fn property_exists_on_edges_and_null_element() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"w\":3}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let val = |q: &str| -> Value {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            run(&plan, &store).rows[0][0].clone()
        };
        // edge carries `w`, not `gone`.
        assert!(matches!(
            val("MATCH ()-[e:R]->() RETURN property_exists(e, w) AS x"),
            Value::Bool(true)
        ));
        assert!(matches!(
            val("MATCH ()-[e:R]->() RETURN property_exists(e, gone) AS x"),
            Value::Bool(false)
        ));
        // OPTIONAL MATCH that finds nothing → m is NULL → property_exists is NULL.
        assert!(val(
            "MATCH (n:N) OPTIONAL MATCH (n)-[:NOSUCH]->(m) RETURN property_exists(m, x) AS x"
        )
        .is_null());
    }

    #[test]
    fn filter_numeric_then_project() {
        let store = social();
        let plan = scan("Person")
            .filter(cmp(CompareOp::Gt, prop(0, "age"), lit(n(28.0))))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "carol"]);
    }

    #[test]
    fn absent_property_is_null_and_filters_as_unknown() {
        let store = social();
        // Project has no age → `age >= 0` is UNKNOWN for it → dropped.
        let plan = Plan::Scan { label: None }
            .filter(cmp(CompareOp::Ge, prop(0, "age"), lit(n(0.0))))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 3);
    }

    #[test]
    fn equality_is_cross_type_false() {
        let store = social();
        let plan = Plan::Scan { label: None }
            .filter(cmp(CompareOp::Eq, prop(0, "age"), lit(s("30"))))
            .project(vec![("name".into(), prop(0, "name"))]);
        assert_eq!(run(&plan, &store).rows.len(), 0);
    }

    /// A hand-built `Insert` plan writes nodes and edges through `execute`.
    #[test]
    fn execute_insert_writes_store() {
        use crate::ir::{InsertEdge, InsertNode};
        let mut store = Builder::default().build();
        let plan = Plan::Insert {
            nodes: vec![
                InsertNode {
                    labels: vec!["P".into()],
                    props: vec![("name".into(), s("a"))],
                },
                InsertNode {
                    labels: vec!["P".into()],
                    props: vec![],
                },
            ],
            edges: vec![InsertEdge {
                from: 0,
                to: 1,
                etype: "R".into(),
                props: vec![],
            }],
        };
        let out = execute(&plan, &mut store).unwrap();
        assert_eq!(out.rows.len(), 0); // a write returns no rows
        assert_eq!(store.node_count(), 2);
        assert_eq!(store.nodes_with_label("P"), &[0, 1]);
        assert_eq!(store.out(0).len(), 1);
        assert_eq!(store.out(0)[0].nbr, 1);
        assert!(matches!(store.prop(0, "name"), Value::Str(x) if &*x == "a"));
    }

    /// A hand-built `Update` plan sets and removes properties on matched nodes.
    /// SET carol.age = 41; REMOVE alice.age — over a Person scan.
    #[test]
    fn execute_update_sets_and_removes() {
        use crate::ir::SetOp;
        let mut store = social();
        let plan = Plan::Update {
            input: Box::new(scan("Person")),
            ops: vec![
                SetOp::Set {
                    slot: 0,
                    key: "seen".into(),
                    value: lit(n(1.0)),
                },
                SetOp::Remove {
                    slot: 0,
                    key: "age".into(),
                },
            ],
        };
        execute(&plan, &mut store).unwrap();
        // every Person got seen=1 and lost age
        for id in 0..3u32 {
            assert!(matches!(store.prop(id, "seen"), Value::Num(x) if x == 1.0));
            assert!(store.prop(id, "age").is_null());
        }
    }

    /// INSERT enforces unique constraints: the second insert of the same key
    /// errors and is rolled back (the graph keeps exactly the first node).
    #[test]
    fn insert_enforces_unique_constraint() {
        use crate::ir::{InsertNode, Plan};
        let mut store = Builder::default().build();
        store.create_unique_constraint("User", &["email"]).unwrap();
        let ins = |email: &str| Plan::Insert {
            nodes: vec![InsertNode {
                labels: vec!["User".into()],
                props: vec![("email".into(), s(email))],
            }],
            edges: vec![],
        };
        assert!(execute(&ins("a@x"), &mut store).is_ok());
        let err = execute(&ins("a@x"), &mut store); // duplicate
        assert!(err.is_err());
        // rolled back: still exactly one User, and node_count did not grow.
        assert_eq!(store.node_count(), 1);
        assert_eq!(store.nodes_with_label("User").len(), 1);
        // a different key still inserts fine.
        assert!(execute(&ins("b@x"), &mut store).is_ok());
        assert_eq!(store.node_count(), 2);
    }

    /// A validator (a per-element `CHECK` predicate) is enforced on write and
    /// rolled back on violation; a null/absent value passes (SQL-`CHECK` semantics).
    #[test]
    fn validator_enforced_on_write() {
        let mut store = Builder::default().build();
        apply_schema_op(
            &mut store,
            r#"{"op":"validator","label":"P","var":"p","predicate":"p.age >= 0"}"#,
        )
        .unwrap();
        let ins = |gql: &str| crate::gql::parse(gql).unwrap();
        assert!(execute(&ins("INSERT (:P {age: 5})"), &mut store).is_ok());
        let err = execute(&ins("INSERT (:P {age: -1})"), &mut store).unwrap_err();
        assert!(err.starts_with("E_VALIDATOR"), "{err}");
        assert_eq!(
            store.nodes_with_label("P").len(),
            1,
            "violating insert rolled back"
        );
        // A null age passes (unknown, not false).
        assert!(execute(&ins("INSERT (:P {name: 'x'})"), &mut store).is_ok());
    }

    /// Declaring a validator the current data already breaks is rejected.
    #[test]
    fn validator_rejects_existing_violation() {
        let mut store = Builder::default().build();
        execute(
            &crate::gql::parse("INSERT (:P {age: -5})").unwrap(),
            &mut store,
        )
        .unwrap();
        let err = apply_schema_op(
            &mut store,
            r#"{"op":"validator","label":"P","var":"p","predicate":"p.age >= 0"}"#,
        );
        assert!(
            matches!(err, Err(crate::schema_op::SchemaError::Rejected(_))),
            "{err:?}"
        );
    }

    /// An invariant (a whole-graph query) is enforced on write; a boolean-`false`
    /// cell in its result rolls the write back.
    #[test]
    fn invariant_enforced_on_write() {
        let mut store = Builder::default().build();
        apply_schema_op(
            &mut store,
            r#"{"op":"invariant","name":"nonneg","query":"MATCH (p:P) RETURN p.age >= 0"}"#,
        )
        .unwrap();
        let ins = |gql: &str| crate::gql::parse(gql).unwrap();
        assert!(execute(&ins("INSERT (:P {age: 1})"), &mut store).is_ok());
        let err = execute(&ins("INSERT (:P {age: -1})"), &mut store).unwrap_err();
        assert!(err.starts_with("E_INVARIANT"), "{err}");
        assert_eq!(
            store.nodes_with_label("P").len(),
            1,
            "violating insert rolled back"
        );
    }

    /// A bad validator predicate / invariant query is a syntax error at declaration.
    #[test]
    fn bad_predicate_and_query_are_syntax_errors() {
        let mut store = Builder::default().build();
        let v = apply_schema_op(
            &mut store,
            r#"{"op":"validator","label":"P","var":"p","predicate":"p.age >=>= 0"}"#,
        );
        assert!(
            matches!(v, Err(crate::schema_op::SchemaError::Syntax(_))),
            "{v:?}"
        );
        let i = apply_schema_op(
            &mut store,
            r#"{"op":"invariant","name":"x","query":"NOT A QUERY"}"#,
        );
        assert!(
            matches!(i, Err(crate::schema_op::SchemaError::Syntax(_))),
            "{i:?}"
        );
    }

    /// A DISTINCT aggregate (other than count) dedups its values per group before
    /// folding — `collect_list(DISTINCT …)`/`min(DISTINCT …)` were folding over
    /// duplicates. Covers the keyless fast-path (which used to skip the dedup) too.
    #[test]
    fn distinct_aggregate_dedups_values() {
        let mut b = Builder::default();
        b.node(&["T"], &[("g", s("a"))]);
        b.node(&["T"], &[("g", s("a"))]);
        b.node(&["T"], &[("g", s("b"))]);
        let st = b.build();
        let list_len = |q: &str| match &run(&crate::gql::parse(q).unwrap(), &st).rows[0][0] {
            Value::List(items) => items.len(),
            o => panic!("expected a list, got {o:?}"),
        };
        // Two distinct `g` values ("a","b"); a constant collapses to one.
        assert_eq!(
            list_len("MATCH (n:T) RETURN collect_list(DISTINCT n.g) AS x"),
            2
        );
        assert_eq!(
            list_len("MATCH (n:T) RETURN collect_list(DISTINCT true) AS x"),
            1
        );
        // Grouped: each group dedups independently (here one group of 2 distinct).
        assert_eq!(
            list_len("MATCH (n:T) RETURN collect_list(DISTINCT n.g) AS x"),
            2
        );
    }

    /// A non-boolean WHERE/FILTER value is a TRUTHINESS test (matching core): a
    /// non-zero number and a non-empty string keep the row; zero / "" / null drop it.
    /// The engine used to treat every non-bool as no-match.
    #[test]
    fn where_rejects_a_non_boolean_condition() {
        let mut b = Builder::default();
        b.node(&["T"], &[("n", n(1.0)), ("s", s("x"))]);
        b.node(&["T"], &[("n", n(0.0)), ("s", s(""))]);
        let st = b.build();
        // A non-boolean WHERE condition is a data exception — a number / string is NOT
        // coerced to a truth value (strict typing; CAST AS BOOLEAN / to_boolean converts).
        let errs = |q: &str| {
            try_run(&crate::gql::parse(q).unwrap(), &st)
                .unwrap_err()
                .contains("E_INVALID_VALUE")
        };
        assert!(errs("MATCH (n:T) WHERE n.n RETURN n.n AS x"));
        assert!(errs("MATCH (n:T) WHERE n.s RETURN n.s AS x"));
        assert!(errs("MATCH (n:T) WHERE 5 RETURN n.n AS x"));
        // A proper boolean condition still works.
        let count = |q: &str| run(&crate::gql::parse(q).unwrap(), &st).rows.len();
        assert_eq!(count("MATCH (n:T) WHERE n.n > 0 RETURN n.n AS x"), 1);
    }

    /// A temporal renders TAGGED in a query result (`{"@duration":"P1D"}`), matching
    /// core — not a bare ISO string. Covers every temporal kind.
    #[test]
    fn query_result_renders_temporals_tagged() {
        let mut b = Builder::default();
        b.node(&["T"], &[]);
        let st = b.build();
        let json = try_run_gql_json(
            &crate::gql::parse(
                "MATCH (n:T) RETURN duration('P1D') AS a, date('2020-01-01') AS b, \
                 local_time('08:30:00') AS c",
            )
            .unwrap(),
            &st,
        )
        .unwrap();
        assert!(json.contains(r#"{"@duration":"P1D"}"#), "{json}");
        assert!(json.contains(r#"{"@date":"2020-01-01"}"#), "{json}");
        assert!(json.contains(r#"{"@localtime":"08:30:00"}"#), "{json}");
    }

    /// INSERT accepts a record/map literal as a property value (a constant record),
    /// stored canonically as a `Value::Record` — the seedable-literal path handles
    /// `{…}`, not just scalars and lists.
    #[test]
    fn insert_writes_a_record_literal_property() {
        let mut store = Builder::default().build();
        execute(
            &crate::gql::parse("INSERT (:P {n: 1, m: {y: 'hi', x: 2}})").unwrap(),
            &mut store,
        )
        .unwrap();
        match store.prop(0, "m") {
            Value::Record(f) => {
                // Canonical: keys sorted (x before y), values preserved.
                assert_eq!(f.len(), 2);
                assert_eq!(f[0].0.as_ref(), "x");
                assert_eq!(format!("{:?}", f[0].1), format!("{:?}", Value::Num(2.0)));
                assert_eq!(f[1].0.as_ref(), "y");
            }
            other => panic!("expected a record, got {other:?}"),
        }
        // A nested field is queryable back out.
        let out = run(
            &crate::gql::parse("MATCH (p:P) RETURN p.m.x AS x").unwrap(),
            &store,
        );
        assert_eq!(
            format!("{:?}", out.rows[0][0]),
            format!("{:?}", Value::Num(2.0))
        );
    }

    /// A single INSERT that creates two colliding nodes is rejected atomically.
    #[test]
    fn insert_rejects_intra_statement_duplicate() {
        use crate::ir::{InsertNode, Plan};
        let mut store = Builder::default().build();
        store.create_unique_constraint("User", &["email"]).unwrap();
        let plan = Plan::Insert {
            nodes: vec![
                InsertNode {
                    labels: vec!["User".into()],
                    props: vec![("email".into(), s("same"))],
                },
                InsertNode {
                    labels: vec!["User".into()],
                    props: vec![("email".into(), s("same"))],
                },
            ],
            edges: vec![],
        };
        assert!(execute(&plan, &mut store).is_err());
        assert_eq!(store.node_count(), 0); // both rolled back
    }

    /// A `SET` that collides with a unique constraint is REJECTED and rolled back —
    /// the Update path enforces constraints like INSERT/_MERGE, not silently apply.
    #[test]
    fn set_enforces_unique_constraint() {
        let mut b = Builder::default();
        b.node(&["User"], &[("email", s("a@x"))]);
        b.node(&["User"], &[("email", s("b@x"))]);
        let mut store = b.build();
        store.create_unique_constraint("User", &["email"]).unwrap();
        let go = |q: &str, store: &mut Store| execute(&crate::gql::parse(q).unwrap(), store);

        // Colliding SET → error, rolled back (still exactly one 'a@x').
        assert!(go(
            "MATCH (u:User) WHERE u.email='b@x' SET u.email='a@x'",
            &mut store
        )
        .is_err());
        let count = |store: &Store, v: &str| {
            store
                .nodes_with_label("User")
                .iter()
                .filter(|&&n| matches!(store.prop(n, "email"), Value::Str(e) if &*e == v))
                .count()
        };
        assert_eq!(count(&store, "a@x"), 1, "collision must have rolled back");
        assert_eq!(count(&store, "b@x"), 1);
        // A non-colliding SET still applies.
        assert!(go(
            "MATCH (u:User) WHERE u.email='b@x' SET u.email='c@x'",
            &mut store
        )
        .is_ok());
        assert_eq!(count(&store, "c@x"), 1);
    }

    // ---- ISO transaction-control keywords (START TRANSACTION / COMMIT / ROLLBACK) ----

    /// Run one GQL statement exactly as `lnk_query`'s GQL path does — through the
    /// shared [`run_query`] dispatcher (transaction control, READ ONLY enforcement,
    /// write/read split), materializing a returned read the way the FFI read path
    /// streams it. So these tests exercise the real integration, not the pieces.
    fn stmt(store: &mut Store, q: &str) -> Result<Rows, String> {
        let plan = crate::gql::parse(q)?;
        match run_query(plan, store)? {
            Executed::Rows(rows) => Ok(rows),
            Executed::Read(plan) => Ok(run(&plan, store)),
        }
    }

    /// Parse `q` and extract the `(kind, read_only)` of the resulting `TxControl`
    /// plan (panicking if it is not one). `Plan` has no `PartialEq`, so the parse
    /// tests compare the extracted parts, not whole plans.
    fn tx_parts(q: &str) -> (TxKind, bool) {
        match crate::gql::parse(q).unwrap_or_else(|e| panic!("parse `{q}`: {e}")) {
            Plan::TxControl { kind, read_only } => (kind, read_only),
            other => panic!("expected TxControl for `{q}`, got {other:?}"),
        }
    }

    #[test]
    fn tx_keywords_parse_to_the_right_plan() {
        assert_eq!(tx_parts("START TRANSACTION"), (TxKind::Start, false));
        assert_eq!(
            tx_parts("START TRANSACTION READ ONLY"),
            (TxKind::Start, true)
        );
        // Case-insensitive; READ WRITE is the (default) read-write mode.
        assert_eq!(
            tx_parts("start transaction read write"),
            (TxKind::Start, false)
        );
        assert_eq!(tx_parts("COMMIT"), (TxKind::Commit, false));
        assert_eq!(tx_parts("COMMIT WORK"), (TxKind::Commit, false));
        assert_eq!(tx_parts("ROLLBACK"), (TxKind::Rollback, false));
        assert_eq!(tx_parts("ROLLBACK WORK"), (TxKind::Rollback, false));
    }

    #[test]
    fn tx_keyword_parse_errors() {
        // START without TRANSACTION, and a bad access mode, are syntax errors.
        assert!(crate::gql::parse("START").is_err());
        assert!(crate::gql::parse("START TRANSACTION READ").is_err());
        assert!(crate::gql::parse("START TRANSACTION READ SOMETIMES").is_err());
        // Trailing input after a complete command is rejected.
        assert!(crate::gql::parse("COMMIT EXTRA").is_err());
    }

    #[test]
    fn commit_keyword_persists_the_transactions_writes() {
        let mut store = Builder::default().build();
        assert!(!store.in_transaction());
        stmt(&mut store, "START TRANSACTION").unwrap();
        assert!(store.in_transaction());
        stmt(&mut store, "INSERT (:Acct {bal: 100})").unwrap();
        stmt(&mut store, "INSERT (:Acct {bal: 200})").unwrap();
        stmt(&mut store, "COMMIT").unwrap();
        assert!(!store.in_transaction(), "COMMIT closes the transaction");
        assert_eq!(store.live_node_count(), 2, "both inserts persisted");
    }

    #[test]
    fn rollback_keyword_discards_the_transactions_writes() {
        let mut store = Builder::default().build();
        stmt(&mut store, "INSERT (:Acct {bal: 1})").unwrap(); // committed implicitly (no tx)
        stmt(&mut store, "START TRANSACTION").unwrap();
        stmt(&mut store, "INSERT (:Acct {bal: 100})").unwrap();
        stmt(&mut store, "INSERT (:Acct {bal: 200})").unwrap();
        stmt(&mut store, "ROLLBACK").unwrap();
        assert!(!store.in_transaction());
        assert_eq!(
            store.live_node_count(),
            1,
            "only the pre-transaction insert survives"
        );
    }

    #[test]
    fn transaction_state_persists_across_separate_statements() {
        // The store IS the session: a START stays open across statement boundaries.
        let mut store = Builder::default().build();
        stmt(&mut store, "START TRANSACTION").unwrap();
        stmt(&mut store, "INSERT (:Acct {bal: 1})").unwrap();
        assert!(store.in_transaction(), "still open between statements");
        stmt(&mut store, "INSERT (:Acct {bal: 2})").unwrap();
        assert!(store.in_transaction());
        stmt(&mut store, "COMMIT").unwrap();
        assert_eq!(store.live_node_count(), 2);
    }

    #[test]
    fn nested_start_transaction_is_a_coded_error() {
        let mut store = Builder::default().build();
        stmt(&mut store, "START TRANSACTION").unwrap();
        let err = stmt(&mut store, "START TRANSACTION").unwrap_err();
        assert!(
            err.starts_with("E_INVALID_GRAPH_OP:"),
            "nested START is E_INVALID_GRAPH_OP, got: {err}"
        );
        assert!(store.in_transaction(), "the original tx is untouched");
        stmt(&mut store, "ROLLBACK").unwrap(); // clean up
    }

    #[test]
    fn commit_or_rollback_with_no_active_transaction_is_a_coded_error() {
        let mut store = Builder::default().build();
        let c = stmt(&mut store, "COMMIT").unwrap_err();
        assert!(c.starts_with("E_INVALID_GRAPH_OP:"), "COMMIT no-tx: {c}");
        let r = stmt(&mut store, "ROLLBACK").unwrap_err();
        assert!(r.starts_with("E_INVALID_GRAPH_OP:"), "ROLLBACK no-tx: {r}");
    }

    #[test]
    fn read_only_transaction_rejects_writes_but_allows_reads() {
        let mut store = Builder::default().build();
        stmt(&mut store, "INSERT (:Acct {bal: 1})").unwrap(); // seed (no tx)
        stmt(&mut store, "START TRANSACTION READ ONLY").unwrap();
        assert!(store.tx_read_only());
        // A read is allowed.
        assert!(stmt(&mut store, "MATCH (n:Acct) RETURN n.bal").is_ok());
        // Every write kind is rejected with the coded error, and nothing changes.
        for w in [
            "INSERT (:Acct {bal: 9})",
            "MATCH (n:Acct) SET n.bal = 5",
            "MATCH (n:Acct) REMOVE n.bal",
            "MATCH (n:Acct) DELETE n",
        ] {
            let e = stmt(&mut store, w).unwrap_err();
            assert!(e.starts_with("E_INVALID_GRAPH_OP:"), "{w} → {e}");
        }
        assert_eq!(
            store.live_node_count(),
            1,
            "read-only left the graph intact"
        );
        stmt(&mut store, "COMMIT").unwrap();
        assert!(!store.tx_read_only(), "COMMIT clears the read-only mode");
        // After commit the mode is cleared — a write applies.
        stmt(&mut store, "INSERT (:Acct {bal: 9})").unwrap();
        assert_eq!(store.live_node_count(), 2);
    }

    #[test]
    fn rollback_clears_the_read_only_mode() {
        let mut store = Builder::default().build();
        stmt(&mut store, "START TRANSACTION READ ONLY").unwrap();
        assert!(store.tx_read_only());
        stmt(&mut store, "ROLLBACK").unwrap();
        assert!(!store.tx_read_only(), "ROLLBACK clears read-only");
        stmt(&mut store, "INSERT (:Acct {bal: 1})").unwrap(); // now allowed
        assert_eq!(store.live_node_count(), 1);
    }

    #[test]
    fn an_immediate_fault_inside_a_transaction_isolates_to_its_own_statement() {
        // A string-`id` collision is an IMMEDIATE fault (the element's identity), so it
        // rolls back only ITS statement's savepoint. The app can swallow it and the
        // writes around it still commit with the transaction — the "skip the bad row,
        // commit the good ones" pattern. (Declared constraints DEFER to commit instead;
        // see the next test.)
        let mut store = Builder::default().build();
        stmt(&mut store, "INSERT (:User {id: 'taken'})").unwrap(); // external id 'taken'
        stmt(&mut store, "START TRANSACTION").unwrap();
        stmt(&mut store, "INSERT (:User {id: 'a'})").unwrap();
        // Collides with the existing 'taken' id → an IMMEDIATE fault; the app ignores it.
        assert!(stmt(&mut store, "INSERT (:User {id: 'taken'})").is_err());
        stmt(&mut store, "INSERT (:User {id: 'b'})").unwrap();
        stmt(&mut store, "COMMIT").unwrap();

        assert!(
            store.node_by_ext("a").is_some(),
            "the write before the fault committed"
        );
        assert!(
            store.node_by_ext("b").is_some(),
            "the write after the fault committed"
        );
        assert!(store.node_by_ext("taken").is_some());
        assert_eq!(store.live_node_count(), 3, "taken + a + b (no duplicate)");
    }

    #[test]
    fn a_deferred_constraint_violation_surfaces_at_commit_and_rolls_the_whole_transaction_back() {
        // A DECLARED unique constraint is checked at COMMIT (deferred), matching core.
        // So the colliding write itself SUCCEEDS mid-transaction, and the violation
        // surfaces only at COMMIT — rolling back the WHOLE transaction (you cannot
        // swallow it row-by-row, unlike an immediate fault).
        let mut b = Builder::default();
        b.node(&["User"], &[("email", s("taken@x"))]);
        let mut store = b.build();
        store.create_unique_constraint("User", &["email"]).unwrap();

        stmt(&mut store, "START TRANSACTION").unwrap();
        stmt(&mut store, "INSERT (:User {email: 'a@x'})").unwrap();
        // Deferred: the duplicate insert itself does NOT fault here.
        stmt(&mut store, "INSERT (:User {email: 'taken@x'})")
            .expect("a deferred unique constraint does not fault at the statement");
        stmt(&mut store, "INSERT (:User {email: 'b@x'})").unwrap();
        let commit = stmt(&mut store, "COMMIT");
        assert!(commit.is_err(), "the deferred violation surfaces at COMMIT");

        let count = |store: &Store, v: &str| {
            store
                .nodes_with_label("User")
                .iter()
                .filter(|&&nd| matches!(store.prop(nd, "email"), Value::Str(e) if &*e == v))
                .count()
        };
        assert_eq!(count(&store, "a@x"), 0, "whole tx rolled back");
        assert_eq!(count(&store, "b@x"), 0, "whole tx rolled back");
        assert_eq!(count(&store, "taken@x"), 1, "only the seed remains");
    }

    #[test]
    fn a_deferred_constraint_completed_by_a_later_statement_commits() {
        // The point of deferral: a row that is temporarily invalid mid-transaction (no
        // required key) becomes valid before COMMIT (a later statement fills it), so the
        // transaction commits — the pattern immediate checking would reject.
        let mut store = Builder::default().build();
        store.create_required_constraint("Acct", "email").unwrap();
        stmt(&mut store, "START TRANSACTION").unwrap();
        // No email yet — deferred, so this SUCCEEDS (immediate checking would reject it).
        stmt(&mut store, "INSERT (:Acct {id: 'u'})")
            .expect("a deferred required constraint does not fault at the statement");
        stmt(
            &mut store,
            "MATCH (n:Acct {id: 'u'}) SET n.email = 'u@x.io'",
        )
        .unwrap();
        stmt(&mut store, "COMMIT").expect("valid by commit time");
        assert_eq!(store.live_node_count(), 1, "the completed row committed");
    }

    #[test]
    fn a_required_violation_in_a_transaction_never_persists() {
        // A required constraint on Acct.email. A row that never fills it must NOT
        // persist — whether the engine rejects it at the statement (its per-statement
        // constraint check) or would defer to COMMIT, the invalid row leaves no trace
        // and the transaction ends cleanly. (Engine checks per-statement; core defers
        // to commit — a separate constraint-deferral divergence — but both are safe.)
        let mut store = Builder::default().build();
        store.create_required_constraint("Acct", "email").unwrap();
        stmt(&mut store, "START TRANSACTION").unwrap();
        let insert = stmt(&mut store, "INSERT (:Acct {bal: 1})"); // no email
        let commit = stmt(&mut store, "COMMIT");
        assert!(
            insert.is_err() || commit.is_err(),
            "a required violation must surface (at the statement or at commit)"
        );
        assert_eq!(store.live_node_count(), 0, "the invalid row left no trace");
        assert!(
            !store.in_transaction(),
            "the transaction is closed either way"
        );
    }

    // ---- row-driven INSERT (`FOR … IN <list> INSERT (…)`) --------------------

    /// Count live nodes carrying label `l`.
    fn label_count(store: &Store, l: &str) -> usize {
        store.nodes_with_label(l).len()
    }

    #[test]
    fn for_insert_parses_to_insert_from() {
        match crate::gql::parse("FOR x IN [1, 2] INSERT (:N {v: x})").unwrap() {
            Plan::InsertFrom { nodes, edges, .. } => {
                assert_eq!(nodes.len(), 1);
                assert_eq!(nodes[0].labels, vec!["N".to_string()]);
                assert_eq!(nodes[0].props.len(), 1);
                assert_eq!(nodes[0].props[0].0, "v");
                assert!(edges.is_empty());
            }
            other => panic!("expected InsertFrom, got {other:?}"),
        }
    }

    #[test]
    fn for_insert_creates_one_node_per_element_with_the_bound_variable() {
        let mut store = Builder::default().build();
        stmt(&mut store, "FOR x IN [1, 2, 3] INSERT (:Acct {bal: x})").unwrap();
        assert_eq!(label_count(&store, "Acct"), 3);
        // Read the values back — one per unwound element.
        let rows = run(
            &crate::gql::parse("MATCH (n:Acct) RETURN n.bal AS bal").unwrap(),
            &store,
        );
        let mut vals: Vec<f64> = rows.rows.iter().map(|r| num(&r[0])).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn for_insert_evaluates_property_expressions_per_row() {
        let mut store = Builder::default().build();
        // `b: x * 2` is an EXPRESSION over the unwound `x`, not a literal.
        stmt(
            &mut store,
            "FOR x IN [10, 20] INSERT (:Pair {a: x, b: x * 2})",
        )
        .unwrap();
        let rows = run(
            &crate::gql::parse("MATCH (n:Pair) RETURN n.a AS a, n.b AS b").unwrap(),
            &store,
        );
        let mut pairs: Vec<(f64, f64)> =
            rows.rows.iter().map(|r| (num(&r[0]), num(&r[1]))).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(pairs, vec![(10.0, 20.0), (20.0, 40.0)]);
    }

    #[test]
    fn for_insert_mixes_literal_and_expression_properties() {
        let mut store = Builder::default().build();
        stmt(
            &mut store,
            "FOR x IN [1, 2] INSERT (:Acct {kind: 'k', bal: x})",
        )
        .unwrap();
        let rows = run(
            &crate::gql::parse("MATCH (n:Acct) RETURN n.kind AS kind, n.bal AS bal").unwrap(),
            &store,
        );
        assert_eq!(rows.rows.len(), 2);
        for r in rows.rows.iter() {
            // the literal `kind` is the same string on every row
            assert!(
                matches!(&r[0], Value::Str(s) if &**s == "k"),
                "kind should be 'k', got {:?}",
                r[0]
            );
        }
    }

    #[test]
    fn for_insert_creates_an_edge_per_row() {
        let mut store = Builder::default().build();
        stmt(
            &mut store,
            "FOR x IN [1, 2] INSERT (:A {v: x})-[:R]->(:B {v: x})",
        )
        .unwrap();
        assert_eq!(label_count(&store, "A"), 2);
        assert_eq!(label_count(&store, "B"), 2);
        // Two R edges, one per row (A_x → B_x).
        let n = run(
            &crate::gql::parse("MATCH (:A)-[:R]->(:B) RETURN count(*) AS c").unwrap(),
            &store,
        );
        assert_eq!(num(&n.rows[0][0]), 2.0);
    }

    #[test]
    fn for_insert_over_an_empty_list_creates_nothing() {
        let mut store = Builder::default().build();
        stmt(&mut store, "FOR x IN [] INSERT (:Acct {bal: x})").unwrap();
        assert_eq!(label_count(&store, "Acct"), 0);
    }

    #[test]
    fn for_insert_is_atomic_a_unique_violation_rolls_back_every_row() {
        // Both rows carry id='dup'; a unique constraint on (Acct, id) means the second
        // collides. Per-statement atomicity must roll the FIRST row back too — zero rows.
        let mut store = Builder::default().build();
        store.create_unique_constraint("Acct", &["id"]).unwrap();
        let err = stmt(
            &mut store,
            "FOR x IN [1, 2] INSERT (:Acct {id: 'dup', bal: x})",
        )
        .unwrap_err();
        assert!(
            err.starts_with("E_UNIQUE:") || err.starts_with("E_CONSTRAINT"),
            "duplicate unique value violates: {err}"
        );
        assert_eq!(
            label_count(&store, "Acct"),
            0,
            "the whole FOR-INSERT rolled back — no partial write"
        );
    }

    #[test]
    fn for_insert_inside_a_transaction_commits_and_rolls_back_as_a_unit() {
        // Committed: the FOR-INSERT's rows persist with the transaction.
        let mut store = Builder::default().build();
        stmt(&mut store, "START TRANSACTION").unwrap();
        stmt(&mut store, "FOR x IN [1, 2, 3] INSERT (:Acct {bal: x})").unwrap();
        stmt(&mut store, "COMMIT").unwrap();
        assert_eq!(label_count(&store, "Acct"), 3);

        // Rolled back: START, FOR-INSERT, ROLLBACK → nothing persists.
        let mut store2 = Builder::default().build();
        stmt(&mut store2, "START TRANSACTION").unwrap();
        stmt(&mut store2, "FOR x IN [1, 2] INSERT (:Acct {bal: x})").unwrap();
        stmt(&mut store2, "ROLLBACK").unwrap();
        assert_eq!(label_count(&store2, "Acct"), 0);
    }

    #[test]
    fn for_insert_is_rejected_in_a_read_only_transaction() {
        let mut store = Builder::default().build();
        stmt(&mut store, "START TRANSACTION READ ONLY").unwrap();
        let err = stmt(&mut store, "FOR x IN [1, 2] INSERT (:Acct {bal: x})").unwrap_err();
        assert!(
            err.starts_with("E_INVALID_GRAPH_OP:"),
            "read-only rejects: {err}"
        );
        assert_eq!(label_count(&store, "Acct"), 0);
    }

    #[test]
    fn insert_string_id_is_the_unique_external_identity() {
        let mut store = Builder::default().build();
        // A string `id` sets the external identity AND stays a queryable property.
        stmt(&mut store, "INSERT (:Acct {id: 'x', bal: 5})").unwrap();
        let rows = run(
            &crate::gql::parse("MATCH (n:Acct) RETURN n.id AS id, n.bal AS bal").unwrap(),
            &store,
        );
        assert!(
            matches!(&rows.rows[0][0], Value::Str(s) if &**s == "x"),
            "n.id stays a stored property"
        );
        assert_eq!(num(&rows.rows[0][1]), 5.0);
        assert!(
            store.node_by_ext("x").is_some(),
            "external id is registered"
        );

        // A duplicate string id is a constraint violation; the graph is unchanged.
        let err = stmt(&mut store, "INSERT (:Acct {id: 'x'})").unwrap_err();
        assert!(err.starts_with("E_UNIQUE:"), "duplicate string id: {err}");
        assert_eq!(label_count(&store, "Acct"), 1);

        // A NUMERIC id is a plain property — no identity, no uniqueness (two coexist).
        stmt(&mut store, "INSERT (:Num {id: 7})").unwrap();
        stmt(&mut store, "INSERT (:Num {id: 7})").unwrap();
        assert_eq!(label_count(&store, "Num"), 2);

        // No id → a synthetic external id; two such nodes coexist.
        stmt(&mut store, "INSERT (:Plain {bal: 1})").unwrap();
        stmt(&mut store, "INSERT (:Plain {bal: 2})").unwrap();
        assert_eq!(label_count(&store, "Plain"), 2);
    }

    #[test]
    fn a_string_id_collision_within_one_insert_rolls_the_whole_statement_back() {
        let mut store = Builder::default().build();
        // Two nodes in ONE INSERT sharing a new string id → the second collides with
        // the first; per-statement atomicity leaves neither.
        let err = stmt(&mut store, "INSERT (:A {id: 'k'}), (:B {id: 'k'})").unwrap_err();
        assert!(
            err.starts_with("E_UNIQUE:"),
            "intra-statement dup id: {err}"
        );
        assert_eq!(store.live_node_count(), 0, "the whole INSERT rolled back");
    }

    #[test]
    fn edge_string_id_is_the_unique_external_identity() {
        let mut store = Builder::default().build();
        stmt(
            &mut store,
            "INSERT (:A {id: 'a'})-[:R {id: 'e1'}]->(:B {id: 'b'})",
        )
        .unwrap();
        // A duplicate EDGE id is a constraint violation; the statement rolls back.
        let err = stmt(
            &mut store,
            "INSERT (:A {id: 'a2'})-[:R {id: 'e1'}]->(:B {id: 'b2'})",
        )
        .unwrap_err();
        assert!(err.starts_with("E_UNIQUE:"), "duplicate edge id: {err}");
        let n = run(
            &crate::gql::parse("MATCH ()-[:R]->() RETURN count(*) AS c").unwrap(),
            &store,
        );
        assert_eq!(num(&n.rows[0][0]), 1.0, "only the first R edge exists");
    }

    #[test]
    fn set_on_a_string_identity_id_is_rejected_but_other_and_numeric_ids_are_settable() {
        let mut store = Builder::default().build();
        stmt(&mut store, "INSERT (:A {id: 'a', bal: 1})").unwrap();
        // SET on a string identity `id` → immutable error; the id is unchanged.
        let err = stmt(&mut store, "MATCH (n:A {id: 'a'}) SET n.id = 'z'").unwrap_err();
        assert!(
            err.starts_with("E_INVALID_GRAPH_OP:"),
            "SET id immutable: {err}"
        );
        assert!(store.node_by_ext("a").is_some(), "the id is unchanged");
        assert!(store.node_by_ext("z").is_none());
        // A NON-id property is still SET-able on an identity node.
        stmt(&mut store, "MATCH (n:A {id: 'a'}) SET n.bal = 5").unwrap();
        // A NUMERIC id is a plain property (not an identity) → SET-able.
        stmt(&mut store, "INSERT (:N {id: 7})").unwrap();
        stmt(&mut store, "MATCH (n:N) SET n.id = 8").unwrap();
    }

    #[test]
    fn set_on_a_string_identity_edge_id_is_rejected() {
        let mut store = Builder::default().build();
        stmt(&mut store, "INSERT (:A)-[:R {id: 'e1', w: 1}]->(:B)").unwrap();
        let err = stmt(&mut store, "MATCH ()-[r:R]->() SET r.id = 'e2'").unwrap_err();
        assert!(
            err.starts_with("E_INVALID_GRAPH_OP:"),
            "SET edge id immutable: {err}"
        );
        // A non-id edge property is still SET-able.
        stmt(&mut store, "MATCH ()-[r:R]->() SET r.w = 9").unwrap();
    }

    /// `REMOVE` of a required-constraint key is rejected and rolled back.
    #[test]
    fn remove_enforces_required_constraint() {
        let mut b = Builder::default();
        b.node(&["User"], &[("name", s("alice"))]);
        let mut store = b.build();
        store.create_required_constraint("User", "name").unwrap();
        let id = store.nodes_with_label("User")[0];
        assert!(execute(
            &crate::gql::parse("MATCH (u:User) REMOVE u.name").unwrap(),
            &mut store
        )
        .is_err());
        assert!(
            store.has_prop(id, "name"),
            "required key must survive rollback"
        );
    }

    /// GQL DELETE / DETACH DELETE, matching core: a non-DETACH delete of a node with
    /// relationships errors and rolls back; DETACH cascades the edges; an edge delete
    /// leaves the endpoints; a node with no edges deletes plainly.
    #[test]
    fn gql_delete_and_detach_delete() {
        let build = || {
            let mut b = Builder::default();
            let a = b.node(&["P"], &[("n", s("a"))]);
            let z = b.node(&["P"], &[("n", s("b"))]);
            let iso = b.node(&["P"], &[("n", s("iso"))]);
            b.edge(a, z, "R");
            let _ = iso;
            b.build()
        };
        let go = |q: &str, store: &mut Store| execute(&crate::gql::parse(q).unwrap(), store);

        // Non-DETACH delete of a node WITH an edge → error, nothing removed.
        let mut s1 = build();
        assert!(go("MATCH (p:P) WHERE p.n='a' DELETE p", &mut s1).is_err());
        assert_eq!(s1.live_node_count(), 3, "rolled back");

        // DETACH DELETE removes the node and its edge; the neighbour survives.
        let mut s2 = build();
        assert!(go("MATCH (p:P) WHERE p.n='a' DETACH DELETE p", &mut s2).is_ok());
        assert_eq!(s2.live_node_count(), 2);

        // A node with NO edges deletes plainly (no DETACH needed).
        let mut s3 = build();
        assert!(go("MATCH (p:P) WHERE p.n='iso' DELETE p", &mut s3).is_ok());
        assert_eq!(s3.live_node_count(), 2);

        // Deleting the EDGE leaves both endpoints; then a plain DELETE works.
        let mut s4 = build();
        assert!(go("MATCH (a:P)-[r:R]->(b) DELETE r", &mut s4).is_ok());
        assert_eq!(s4.live_node_count(), 3);
        assert!(go("MATCH (p:P) WHERE p.n='a' DELETE p", &mut s4).is_ok());
        assert_eq!(s4.live_node_count(), 2);
    }

    /// A deleted node is absent from a label scan through the query path — build
    /// the social graph, delete bob (id 1), and the Person scan yields alice+carol.
    #[test]
    fn scan_skips_deleted_node() {
        let mut store = social();
        store.delete_node(1); // bob
        let out = run(
            &scan("Person").project(vec![("name".into(), prop(0, "name"))]),
            &store,
        );
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "carol"]);
    }

    // --- Arithmetic (E1) ---

    fn arith(op: crate::ir::ArithOp, l: Expr, r: Expr) -> Expr {
        Expr::Arith {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }

    /// `age * 2 + 1` for alice(30) = 61 — precedence honored in the hand plan.
    #[test]
    fn arith_eval_computes() {
        use crate::ir::ArithOp::{Add, Mul};
        let store = social();
        let plan = scan("Person")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))))
            .project(vec![(
                "x".into(),
                arith(Add, arith(Mul, prop(0, "age"), lit(n(2.0))), lit(n(1.0))),
            )]);
        assert_eq!(num(&run(&plan, &store).rows[0][0]), 61.0);
    }

    /// A NULL / missing / non-numeric operand yields NULL — the Project node has no
    /// `age`, so `age + 1` is NULL for exactly it.
    #[test]
    fn arith_null_propagates() {
        use crate::ir::ArithOp::Add;
        let store = social();
        let plan = Plan::Scan { label: None }
            .project(vec![("x".into(), arith(Add, prop(0, "age"), lit(n(1.0))))]);
        let nulls = run(&plan, &store)
            .rows
            .iter()
            .filter(|r| r[0].is_null())
            .count();
        assert_eq!(nulls, 1); // only the Project node lacks age
    }

    /// Arithmetic follows core's SQL rule: a NULL operand yields NULL, but a non-null
    /// NON-numeric operand (string/bool) is a DATA EXCEPTION (never coerced) — an
    /// explicit CAST is the escape hatch. Aggregates sum()/avg() likewise throw over a
    /// non-numeric value.
    #[test]
    fn arith_and_agg_throw_on_non_numeric() {
        let store = social();
        let ok = |q: &str| {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            try_run(&plan, &store)
        };
        // null operand → NULL (not an error).
        assert!(ok("RETURN 1 + null AS r").unwrap().rows[0][0].is_null());
        // non-null non-numeric → error.
        assert!(ok("RETURN 'abc' + 1 AS r").is_err());
        assert!(ok("RETURN true * 2 AS r").is_err());
        assert!(ok("MATCH (p:Person) RETURN p.name + 1 AS r").is_err());
        // CAST is the escape hatch.
        assert!(matches!(
            ok("RETURN CAST('2' AS INT) * 3 AS r").unwrap().rows[0][0],
            Value::Num(x) if x == 6.0
        ));
        // sum/avg over numbers still work; over a non-numeric they throw.
        assert!(ok("MATCH (p:Person) RETURN sum(p.age) AS r").is_ok());
        assert!(ok("MATCH (p:Person) RETURN sum(p.name) AS r").is_err());
        assert!(ok("MATCH (p:Person) RETURN avg(p.name) AS r").is_err());
    }

    /// Division / modulo by zero THROWS (matches lenke-core's DataException), via
    /// the fallible read path — `try_run` surfaces the error (K3).
    #[test]
    fn arith_div_or_mod_by_zero_throws() {
        use crate::ir::ArithOp::{Div, Rem};
        let store = social();
        let one = scan("Person").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))));
        for op in [Div, Rem] {
            let plan = one
                .clone()
                .project(vec![("x".into(), arith(op, prop(0, "age"), lit(n(0.0))))]);
            let err = crate::exec::try_run(&plan, &store).unwrap_err();
            assert!(err.contains("division by zero"), "op {op:?}: {err}");
        }
    }

    /// A product that overflows f64 to +Inf is KEPT (IEEE), matching lenke-core —
    /// NaN/Inf are coerced to null only at the JSON egress boundary, not here (K4).
    #[test]
    fn arith_overflow_keeps_inf() {
        use crate::ir::ArithOp::Mul;
        let store = social();
        let one = scan("Person").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))));
        let big = one.project(vec![("x".into(), arith(Mul, lit(n(1e308)), lit(n(1e308))))]);
        assert!(
            matches!(run(&big, &store).rows[0][0], Value::Num(x) if x.is_infinite() && x > 0.0)
        );
    }

    // --- Property index + IndexSeek (D1a) ---

    /// A store with two labels sharing an `age` property (some age 30).
    fn indexed_store() -> Store {
        let mut st = Builder::default().build();
        st.add_node(&["P"], &[("age", n(30.0)), ("name", s("a"))]);
        st.add_node(&["P"], &[("age", n(25.0)), ("name", s("b"))]);
        st.add_node(&["P"], &[("age", n(30.0)), ("name", s("c"))]);
        st.add_node(&["Q"], &[("age", n(30.0)), ("name", s("d"))]); // other label
        st
    }

    /// `IndexSeek` returns the SAME rows as `Scan + Filter(=)`, with and without
    /// an index. P nodes with age 30 are a and c (d is a Q, excluded).
    #[test]
    fn index_seek_matches_scan_filter() {
        let mut st = indexed_store();
        let seek = Plan::IndexSeek {
            label: "P".into(),
            key: "age".into(),
            value: n(30.0),
        }
        .project(vec![("name".into(), prop(0, "name"))]);
        let filt = scan("P")
            .filter(cmp(CompareOp::Eq, prop(0, "age"), lit(n(30.0))))
            .project(vec![("name".into(), prop(0, "name"))]);

        let mut want = names_of(&run(&filt, &st), 0);
        want.sort();
        assert_eq!(want, vec!["a", "c"]);
        let mut got = names_of(&run(&seek, &st), 0);
        got.sort();
        assert_eq!(got, want); // no index yet (scan fallback)

        st.create_index("age");
        let mut got = names_of(&run(&seek, &st), 0);
        got.sort();
        assert_eq!(got, want); // index path, same rows
    }

    /// The index is maintained through set/remove/delete.
    #[test]
    fn index_maintained_on_writes() {
        let mut st = indexed_store();
        st.create_index("age");
        let sorted = |st: &Store| {
            let mut v = st.index_lookup("age", &n(30.0)).unwrap();
            v.sort_unstable();
            v
        };
        assert_eq!(sorted(&st), vec![0, 2, 3]); // any-label candidates
        st.set_prop(0, "age", n(25.0)); // 0 leaves the 30 bucket
        assert_eq!(sorted(&st), vec![2, 3]);
        st.delete_node(2); // 2 gone
        assert_eq!(sorted(&st), vec![3]);
        st.remove_prop(3, "age"); // 3 loses the prop
        assert!(st.index_lookup("age", &n(30.0)).unwrap().is_empty());
    }

    /// A transaction rollback restores the index (writes replay through the
    /// primitives, which maintain it).
    #[test]
    fn index_consistent_after_rollback() {
        let mut st = indexed_store();
        st.create_index("age");
        st.begin();
        st.set_prop(0, "age", n(99.0));
        st.delete_node(2);
        st.rollback();
        let mut v = st.index_lookup("age", &n(30.0)).unwrap();
        v.sort_unstable();
        assert_eq!(v, vec![0, 2, 3]);
    }

    /// A NaN / NULL seek value matches nothing (predicate `=` semantics), same as
    /// the filter — even though those values live in a group_key bucket.
    #[test]
    fn index_seek_nan_and_null_match_nothing() {
        let mut st = indexed_store();
        st.create_index("age");
        let seek = |v: Value| {
            Plan::IndexSeek {
                label: "P".into(),
                key: "age".into(),
                value: v,
            }
            .project(vec![("name".into(), prop(0, "name"))])
        };
        assert_eq!(run(&seek(n(f64::NAN)), &st).rows.len(), 0);
        assert_eq!(run(&seek(Value::Null), &st).rows.len(), 0);
    }

    /// `RangeSeek` returns the SAME rows as `Scan + Filter(<op>)` for every range
    /// op, with and without a range index. Hand: ages 30,25,40 (a,b,c).
    #[test]
    fn range_seek_matches_scan_filter_all_ops() {
        let mut st = indexed_store(); // P: a=30, b=25, c=30; Q: d=30
        let ops = [
            (CompareOp::Gt, 25.0, vec!["a", "c"]), // >25 → 30,30
            (CompareOp::Ge, 30.0, vec!["a", "c"]), // >=30
            (CompareOp::Lt, 30.0, vec!["b"]),      // <30 → 25
            (CompareOp::Le, 25.0, vec!["b"]),      // <=25
        ];
        for indexed in [false, true] {
            if indexed {
                st.create_range_index("age");
            }
            for (op, v, want) in &ops {
                let seek = Plan::RangeSeek {
                    label: "P".into(),
                    key: "age".into(),
                    op: *op,
                    value: n(*v),
                }
                .project(vec![("name".into(), prop(0, "name"))]);
                let filt = scan("P")
                    .filter(cmp(*op, prop(0, "age"), lit(n(*v))))
                    .project(vec![("name".into(), prop(0, "name"))]);
                let mut a = names_of(&run(&seek, &st), 0);
                a.sort();
                let mut b = names_of(&run(&filt, &st), 0);
                b.sort();
                assert_eq!(a, *want, "op {op:?} v {v}");
                assert_eq!(a, b, "seek vs filter disagree for {op:?} {v}");
            }
        }
    }

    /// An indexed range seek returns EXACTLY the scan-filter rows (the
    /// equivalent-spellings invariant): a NULL value matches nothing, and a
    /// cross-type comparison is UNKNOWN → dropped (a string property vs a numeric
    /// bound does NOT match, per the 3VL operator semantics — K2).
    #[test]
    fn range_seek_null_and_cross_type_match_filter() {
        let mut st = Builder::default().build();
        st.add_node(&["P"], &[("v", n(10.0))]);
        st.add_node(&["P"], &[("v", s("zzz"))]); // string: cross-type vs a number
        st.add_node(&["P"], &[]); // v absent → null
        st.create_range_index("v");
        let check = |st: &Store, op, val: Value| {
            let seek = Plan::RangeSeek {
                label: "P".into(),
                key: "v".into(),
                op,
                value: val.clone(),
            };
            let filt = scan("P").filter(cmp(op, prop(0, "v"), lit(val)));
            (run(&seek, st).rows.len(), run(&filt, st).rows.len())
        };
        // v > 5: only 10 matches; "zzz" is cross-type → UNKNOWN → dropped; null
        // excluded → 1, and seek agrees with filter.
        assert_eq!(check(&st, CompareOp::Gt, n(5.0)), (1, 1));
        // v > null: UNKNOWN for all → 0, agree.
        assert_eq!(check(&st, CompareOp::Gt, Value::Null), (0, 0));
    }

    /// The range index is maintained through set/delete and a transaction rollback.
    #[test]
    fn range_index_maintained_and_rolls_back() {
        let mut st = indexed_store();
        st.create_range_index("age");
        // Candidates > 25 across any label (index_lookup is any-label).
        let cand = |st: &Store, v: f64| st.range_lookup("age", CompareOp::Gt, &n(v)).unwrap().len();
        assert_eq!(cand(&st, 25.0), 3); // a,c (P,30) + d (Q,30)
        st.set_prop(0, "age", n(10.0)); // a drops below 25
        assert_eq!(cand(&st, 25.0), 2);
        st.begin();
        st.delete_node(2); // c gone
        assert_eq!(cand(&st, 25.0), 1);
        st.rollback();
        assert_eq!(cand(&st, 25.0), 2); // restored
    }

    /// A scalar count over an IndexSeek is correct (the seek seeds like a scan).
    #[test]
    fn count_over_index_seek() {
        let mut st = indexed_store();
        st.create_index("age");
        let plan = Plan::IndexSeek {
            label: "P".into(),
            key: "age".into(),
            value: n(30.0),
        }
        .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        assert_eq!(num(&run(&plan, &st).rows[0][0]), 2.0);
    }

    /// Reversed operand order (`literal < prop`) must match `prop > literal` —
    /// exercises the fused filter's operand flip. `28 < age` → alice(30),carol(40).
    #[test]
    fn filter_literal_on_left_flips() {
        let store = social();
        let plan = scan("Person")
            .filter(cmp(CompareOp::Lt, lit(n(28.0)), prop(0, "age")))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "carol"]);
    }

    // --- Expand ---

    /// `MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name` — two slots bound,
    /// row per matching edge.
    #[test]
    fn expand_binds_both_ends() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .project(vec![
                ("a".into(), prop(0, "name")),
                ("b".into(), prop(1, "name")),
            ]);
        let out = run(&plan, &store);
        let mut pairs: Vec<(String, String)> = out
            .rows
            .iter()
            .map(|r| (as_str(&r[0]), as_str(&r[1])))
            .collect();
        pairs.sort();
        // a→b, a→c, b→c (KNOWS only; the WORKS_ON edge is excluded)
        assert_eq!(
            pairs,
            vec![
                ("alice".into(), "bob".into()),
                ("alice".into(), "carol".into()),
                ("bob".into(), "carol".into()),
            ]
        );
    }

    /// An edge-label filter selects: WORKS_ON reaches only the Project.
    #[test]
    fn expand_filters_by_edge_label() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["WORKS_ON".to_string()])
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["graphdb"]);
    }

    /// Filtering on the FAR end after an expand — the far slot's property.
    #[test]
    fn filter_on_the_expanded_end() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .filter(cmp(CompareOp::Ge, prop(1, "age"), lit(n(40.0))))
            .project(vec![("a".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // Only edges landing on carol(40): alice→carol, bob→carol.
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "bob"]);
    }

    /// Incoming direction: who KNOWS carol.
    #[test]
    fn expand_incoming() {
        let store = social();
        let plan = scan("Person")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("carol"))))
            .expand(0, Dir::In, &["KNOWS".to_string()])
            .project(vec![("who".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "bob"]);
    }

    /// An unknown edge label matches nothing.
    #[test]
    fn expand_unknown_label_is_empty() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["NOPE".to_string()])
            .project(vec![("x".into(), prop(1, "name"))]);
        assert_eq!(run(&plan, &store).rows.len(), 0);
    }

    /// `expand_edge` binds the traversed edge as a slot: for `(a)-[r:R]->(b)` the
    /// edge is slot 1 and the node slot 2, so `r.weight` reads an edge property and
    /// `b.name` reads a node property.
    #[test]
    fn expand_edge_binds_edge_and_reads_edge_prop() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        st.add_edge(a, b, "R");
        let eid = st.out(a)[0].eid;
        st.set_edge_prop(eid, "weight", n(0.5));
        let plan = scan("P")
            .expand_edge(0, Dir::Out, &["R".to_string()])
            .project(vec![
                ("w".into(), prop(1, "weight")), // edge slot
                ("b".into(), prop(2, "name")),   // node slot
            ]);
        let out = run(&plan, &st);
        assert_eq!(out.rows.len(), 1);
        assert!(matches!(&out.rows[0][0], Value::Num(x) if *x == 0.5));
        assert_eq!(as_str(&out.rows[0][1]), "b");
    }

    /// An edge slot with no such property reads NULL.
    #[test]
    fn expand_edge_absent_prop_is_null() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[]);
        let b = st.add_node(&["P"], &[]);
        st.add_edge(a, b, "R");
        let plan = scan("P")
            .expand_edge(0, Dir::Out, &["R".to_string()])
            .project(vec![("w".into(), prop(1, "weight"))]);
        let out = run(&plan, &st);
        assert_eq!(out.rows.len(), 1);
        assert!(out.rows[0][0].is_null());
    }

    /// Filtering on an edge property keeps only matching edges. a→b (w=0.5),
    /// a→c (w=0.2); `WHERE r.w > 0.4` → only b.
    #[test]
    fn filter_on_edge_property() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        let c = st.add_node(&["P"], &[("name", s("c"))]);
        st.add_edge(a, b, "R");
        let e1 = st.out(a)[0].eid;
        st.add_edge(a, c, "R");
        let e2 = st.out(a)[1].eid;
        st.set_edge_prop(e1, "w", n(0.5));
        st.set_edge_prop(e2, "w", n(0.2));
        let plan = scan("P")
            .expand_edge(0, Dir::Out, &["R".to_string()])
            .filter(cmp(CompareOp::Gt, prop(1, "w"), lit(n(0.4))))
            .project(vec![("b".into(), prop(2, "name"))]);
        assert_eq!(names_of(&run(&plan, &st), 0), vec!["b"]);
    }

    fn as_str(v: &Value) -> String {
        match v {
            Value::Str(x) => x.to_string(),
            other => format!("{other:?}"),
        }
    }

    fn num(v: &Value) -> f64 {
        match v {
            Value::Num(x) => *x,
            other => panic!("expected a number, got {other:?}"),
        }
    }

    // --- Aggregate / group-by ---

    fn agg(func: AggFn, arg: Option<Expr>, distinct: bool, name: &str) -> Agg {
        Agg {
            func,
            arg,
            distinct,
            name: name.to_string(),
            frac: None,
            null_on_empty: false,
        }
    }

    /// Scalar `count(*)` over a label — one row, the count.
    #[test]
    fn scalar_count_star() {
        let store = social();
        let plan = scan("Person").aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        let out = run(&plan, &store);
        assert_eq!(out.names, vec!["c"]);
        assert_eq!(out.rows.len(), 1);
        assert_eq!(num(&out.rows[0][0]), 3.0); // alice, bob, carol
    }

    /// sum / min / max / avg over the age column, hand-computed: 30,25,40.
    #[test]
    fn scalar_sum_min_max_avg() {
        let store = social();
        let plan = scan("Person").aggregate(
            vec![],
            vec![
                agg(AggFn::Sum, Some(prop(0, "age")), false, "s"),
                agg(AggFn::Min, Some(prop(0, "age")), false, "lo"),
                agg(AggFn::Max, Some(prop(0, "age")), false, "hi"),
                agg(AggFn::Avg, Some(prop(0, "age")), false, "av"),
            ],
        );
        let out = run(&plan, &store);
        let r = &out.rows[0];
        assert_eq!(num(&r[0]), 95.0); // 30+25+40
        assert_eq!(num(&r[1]), 25.0);
        assert_eq!(num(&r[2]), 40.0);
        assert_eq!(num(&r[3]), 95.0 / 3.0);
    }

    /// `count(*)` grouped by a property — a row per distinct value, first-seen
    /// order. Group on `city`: alice/carol="nyc", bob="sf".
    #[test]
    fn group_count_by_property() {
        let mut b = Builder::default();
        b.node(&["P"], &[("city", s("nyc"))]);
        b.node(&["P"], &[("city", s("sf"))]);
        b.node(&["P"], &[("city", s("nyc"))]);
        let store = b.build();
        let plan = scan("P").aggregate(
            vec![("city".into(), prop(0, "city"))],
            vec![agg(AggFn::Count, None, false, "c")],
        );
        let out = run(&plan, &store);
        assert_eq!(out.names, vec!["city", "c"]);
        // first-seen order: nyc (row 0), then sf (row 1).
        assert_eq!(as_str(&out.rows[0][0]), "nyc");
        assert_eq!(num(&out.rows[0][1]), 2.0);
        assert_eq!(as_str(&out.rows[1][0]), "sf");
        assert_eq!(num(&out.rows[1][1]), 1.0);
    }

    /// `count(arg)` ignores nulls; `count(DISTINCT arg)` ignores nulls AND
    /// duplicates. Ages: 10, 10, null, 20 → count=3, distinct=2.
    #[test]
    fn count_arg_and_count_distinct_skip_nulls() {
        let mut b = Builder::default();
        b.node(&["P"], &[("v", n(10.0))]);
        b.node(&["P"], &[("v", n(10.0))]);
        b.node(&["P"], &[]); // no v → null
        b.node(&["P"], &[("v", n(20.0))]);
        let store = b.build();
        let plan = scan("P").aggregate(
            vec![],
            vec![
                agg(AggFn::Count, Some(prop(0, "v")), false, "c"),
                agg(AggFn::Count, Some(prop(0, "v")), true, "cd"),
            ],
        );
        let out = run(&plan, &store);
        assert_eq!(num(&out.rows[0][0]), 3.0); // non-null count
        assert_eq!(num(&out.rows[0][1]), 2.0); // distinct non-null: {10, 20}
    }

    /// Over nothing, `count` and `sum` are both 0 but `avg` is NULL — matching
    /// lenke-core (the GQL/Cypher convention; the differential fuzzer flagged the
    /// earlier SQL-style `sum → NULL`).
    #[test]
    fn sum_over_empty_is_zero_avg_is_null() {
        let store = social();
        // No node has this label → empty input to the scalar aggregate.
        let plan = Plan::Scan {
            label: Some("Nonexistent".into()),
        }
        .aggregate(
            vec![],
            vec![
                agg(AggFn::Count, None, false, "c"),
                agg(AggFn::Sum, Some(prop(0, "age")), false, "s"),
                agg(AggFn::Avg, Some(prop(0, "age")), false, "a"),
            ],
        );
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1); // scalar aggregate still emits one row
        assert_eq!(num(&out.rows[0][0]), 0.0); // count(*) = 0
        assert_eq!(num(&out.rows[0][1]), 0.0); // sum = 0
        assert!(out.rows[0][2].is_null()); // avg = NULL
    }

    /// A grouped aggregate over empty input emits ZERO rows (unlike the scalar
    /// case) — there are no groups.
    #[test]
    fn grouped_over_empty_is_zero_rows() {
        let store = social();
        let plan = Plan::Scan {
            label: Some("Nonexistent".into()),
        }
        .aggregate(
            vec![("k".into(), prop(0, "age"))],
            vec![agg(AggFn::Count, None, false, "c")],
        );
        assert_eq!(run(&plan, &store).rows.len(), 0);
    }

    /// Aggregate after an Expand: out-degree per person (count of KNOWS edges),
    /// grouped by the source. alice→2, bob→1, carol→0(absent).
    #[test]
    fn count_out_degree_grouped_by_source() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .aggregate(
                vec![("who".into(), prop(0, "name"))],
                vec![agg(AggFn::Count, None, false, "deg")],
            );
        let out = run(&plan, &store);
        let mut got: Vec<(String, f64)> = out
            .rows
            .iter()
            .map(|r| (as_str(&r[0]), num(&r[1])))
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        // carol has no outgoing KNOWS, so she is absent from the expanded rows.
        assert_eq!(got, vec![("alice".into(), 2.0), ("bob".into(), 1.0)]);
    }

    /// Scalar `count(*)` over a single Expand — the frontier fast path. Hand
    /// count of KNOWS edges: alice→{bob,carol}, bob→{carol} = 3.
    #[test]
    fn fused_count_star_one_hop() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1);
        assert_eq!(num(&out.rows[0][0]), 3.0);
    }

    /// Scalar `count(*)` over a two-hop chain. Hand count of length-2 KNOWS
    /// walks: only alice→bob→carol (bob is the only reached node with an
    /// outgoing KNOWS) = 1.
    #[test]
    fn fused_count_star_two_hop() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .expand(1, Dir::Out, &["KNOWS".to_string()])
            .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        let out = run(&plan, &store);
        assert_eq!(num(&out.rows[0][0]), 1.0);
    }

    /// 2-hop `count(*)` where an intermediate is reached by MULTIPLE paths — the
    /// dedup-with-multiplicity path must scale by how many times it was reached.
    /// a→x, b→x (x reached twice), x→p, x→q. Length-2 walks: a→x→{p,q} and
    /// b→x→{p,q} = 4 (x itself reaches p,q which are sinks).
    #[test]
    fn fused_count_star_two_hop_with_multiplicity() {
        let mut bld = Builder::default();
        let a = bld.node(&["P"], &[]);
        let b = bld.node(&["P"], &[]);
        let x = bld.node(&["P"], &[]);
        let p = bld.node(&["P"], &[]);
        let q = bld.node(&["P"], &[]);
        bld.edge(a, x, "R");
        bld.edge(b, x, "R");
        bld.edge(x, p, "R");
        bld.edge(x, q, "R");
        let store = bld.build();
        let plan = scan("P")
            .expand(0, Dir::Out, &["R".to_string()])
            .expand(1, Dir::Out, &["R".to_string()])
            .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        assert_eq!(num(&run(&plan, &store).rows[0][0]), 4.0);
    }

    /// `count(DISTINCT c)` over the two-hop chain: the distinct endpoints are
    /// {carol} = 1, deduped in the bitset path.
    #[test]
    fn fused_count_distinct_endpoint() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .expand(1, Dir::Out, &["KNOWS".to_string()])
            .aggregate(
                vec![],
                vec![agg(AggFn::Count, Some(Expr::Slot(2)), true, "c")],
            );
        let out = run(&plan, &store);
        assert_eq!(num(&out.rows[0][0]), 1.0);
    }

    /// Grouped count over an Expand chain (the frontier-mode aggregate). Group
    /// the reached KNOWS neighbours by name: alice→{bob,carol}, bob→{carol}, so
    /// the frontier is [bob, carol, carol] → bob:1, carol:2, first-seen order.
    #[test]
    fn frontier_grouped_count_matches() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .aggregate(
                vec![("who".into(), prop(1, "name"))],
                vec![agg(AggFn::Count, None, false, "c")],
            );
        let out = run(&plan, &store);
        assert_eq!(as_str(&out.rows[0][0]), "bob");
        assert_eq!(num(&out.rows[0][1]), 1.0);
        assert_eq!(as_str(&out.rows[1][0]), "carol");
        assert_eq!(num(&out.rows[1][1]), 2.0);
    }

    /// The node-grouped count path when DISTINCT nodes share a property value —
    /// the level-2 merge must combine them. a→{b,c,d}; b,d are in nyc, c in sf.
    /// Group reached neighbours by city: nyc:2 (b,d), sf:1, first-seen order.
    #[test]
    fn node_grouped_count_merges_shared_value() {
        let mut b = Builder::default();
        let a = b.node(&["P"], &[("name", s("a"))]);
        let n1 = b.node(&["P"], &[("city", s("nyc"))]);
        let n2 = b.node(&["P"], &[("city", s("sf"))]);
        let n3 = b.node(&["P"], &[("city", s("nyc"))]);
        b.edge(a, n1, "R");
        b.edge(a, n2, "R");
        b.edge(a, n3, "R");
        let store = b.build();
        let plan = scan("P").expand(0, Dir::Out, &["R".to_string()]).aggregate(
            vec![("city".into(), prop(1, "city"))],
            vec![agg(AggFn::Count, None, false, "c")],
        );
        let out = run(&plan, &store);
        assert_eq!(as_str(&out.rows[0][0]), "nyc");
        assert_eq!(num(&out.rows[0][1]), 2.0);
        assert_eq!(as_str(&out.rows[1][0]), "sf");
        assert_eq!(num(&out.rows[1][1]), 1.0);
    }

    /// A grouped SUM over the frontier's property, to exercise a non-count agg on
    /// the frontier path: sum the neighbours' ages by name. bob(25) reached once;
    /// carol(40) reached twice → 80.
    #[test]
    fn frontier_grouped_sum_matches() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .aggregate(
                vec![("who".into(), prop(1, "name"))],
                vec![agg(AggFn::Sum, Some(prop(1, "age")), false, "s")],
            );
        let out = run(&plan, &store);
        assert_eq!(num(&out.rows[0][1]), 25.0); // bob
        assert_eq!(num(&out.rows[1][1]), 80.0); // carol twice
    }

    /// An unknown final edge label fuses to zero rows.
    #[test]
    fn fused_count_unknown_label_is_zero() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["NOPE".to_string()])
            .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        assert_eq!(num(&run(&plan, &store).rows[0][0]), 0.0);
    }

    // --- Order + Page ---

    fn asc(slot: usize, key: &str) -> crate::ir::SortKey {
        crate::ir::SortKey {
            expr: prop(slot, key),
            descending: false,
            nulls_first: false,
        }
    }
    fn desc(slot: usize, key: &str) -> crate::ir::SortKey {
        crate::ir::SortKey {
            expr: prop(slot, key),
            descending: true,
            nulls_first: false,
        }
    }

    /// NULL placement is a language contract independent of direction: GQL keeps
    /// NULLs LAST in both ASC and DESC (a NULL prop must not float to the front
    /// under DESC). Uses a graph where one node lacks `age`.
    #[test]
    fn gql_order_by_desc_keeps_nulls_last() {
        let mut b = Builder::default();
        b.node(&["P"], &[("age", n(30.0))]);
        b.node(&["P"], &[("age", n(10.0))]);
        b.node(&["P"], &[]); // no age → NULL
        let store = b.build();
        let ages =
            |q: &str| -> Vec<String> { names_of(&run(&crate::gql::parse(q).unwrap(), &store), 1) };
        // DESC: 30, 10, then NULL last (not first).
        assert_eq!(
            ages("MATCH (p:P) RETURN p.age AS a0, p.age AS a1 ORDER BY a0 DESC"),
            vec!["Num(30.0)", "Num(10.0)", "Null"]
        );
        // ASC: 10, 30, NULL last.
        assert_eq!(
            ages("MATCH (p:P) RETURN p.age AS a0, p.age AS a1 ORDER BY a0 ASC"),
            vec!["Num(10.0)", "Num(30.0)", "Null"]
        );
    }

    /// Gremlin's `order()` places NULLs FIRST (the other language default) — the
    /// same shared OrderPage, driven by `SortKey.nulls_first`.
    #[test]
    fn gremlin_order_keeps_nulls_first() {
        let mut b = Builder::default();
        b.node(&["P"], &[("age", n(30.0)), ("name", s("a"))]);
        b.node(&["P"], &[("age", n(10.0)), ("name", s("b"))]);
        b.node(&["P"], &[("name", s("c"))]); // no age → NULL
        let store = b.build();
        let out = run(
            &crate::gremlin::parse("g.V().hasLabel('P').order().by('age').values('name')").unwrap(),
            &store,
        );
        // NULL-age node ('c') sorts FIRST, then 10 ('b'), 30 ('a').
        assert_eq!(names_of(&out, 0), vec!["c", "b", "a"]);
    }

    /// ORDER BY age ascending, then project name.
    #[test]
    fn order_by_ascending() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![asc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // ages 30,25,40 -> bob(25), alice(30), carol(40)
        assert_eq!(names_of(&out, 0), vec!["bob", "alice", "carol"]);
    }

    /// Descending reverses it.
    #[test]
    fn order_by_descending() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![desc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["carol", "alice", "bob"]);
    }

    /// ORDER BY ... LIMIT is a top-k prefix of the sorted order.
    #[test]
    fn order_then_limit_is_top_k() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![desc(0, "age")], None, Some(2))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["carol", "alice"]); // two oldest
    }

    /// SKIP then LIMIT is a paging window over the sorted order.
    #[test]
    fn order_skip_limit_paging_window() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![asc(0, "age")], Some(1), Some(1))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // sorted bob,alice,carol; skip 1, take 1 -> alice
        assert_eq!(names_of(&out, 0), vec!["alice"]);
    }

    /// Nulls sort LAST in ascending order (the value contract's policy).
    #[test]
    fn nulls_sort_last_ascending() {
        let mut b = Builder::default();
        b.node(&["P"], &[("name", s("has30")), ("age", n(30.0))]);
        b.node(&["P"], &[("name", s("noage"))]); // null age
        b.node(&["P"], &[("name", s("has10")), ("age", n(10.0))]);
        let store = b.build();
        let plan = scan("P")
            .order_page(vec![asc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // 10, 30, then null last
        assert_eq!(names_of(&out, 0), vec!["has10", "has30", "noage"]);
    }

    /// Multi-key: city ascending, then age descending within a city.
    #[test]
    fn multi_key_order() {
        let mut b = Builder::default();
        b.node(
            &["P"],
            &[("name", s("a")), ("city", s("nyc")), ("age", n(30.0))],
        );
        b.node(
            &["P"],
            &[("name", s("b")), ("city", s("sf")), ("age", n(40.0))],
        );
        b.node(
            &["P"],
            &[("name", s("c")), ("city", s("nyc")), ("age", n(50.0))],
        );
        let store = b.build();
        let plan = scan("P")
            .order_page(vec![asc(0, "city"), desc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // nyc: c(50) before a(30); then sf: b(40)
        assert_eq!(names_of(&out, 0), vec!["c", "a", "b"]);
    }

    // --- Distinct ---

    /// `RETURN DISTINCT city` over nyc/sf/nyc -> two rows, first-seen order.
    #[test]
    fn distinct_dedups_projected_column() {
        let mut b = Builder::default();
        b.node(&["P"], &[("city", s("nyc"))]);
        b.node(&["P"], &[("city", s("sf"))]);
        b.node(&["P"], &[("city", s("nyc"))]);
        let store = b.build();
        let plan = scan("P")
            .project(vec![("city".into(), prop(0, "city"))])
            .distinct();
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["nyc", "sf"]);
    }

    /// DISTINCT is over the WHOLE projected row: (city, tier) tuples dedup, so a
    /// repeated city with a different tier is NOT collapsed.
    #[test]
    fn distinct_is_over_the_whole_row() {
        let mut b = Builder::default();
        b.node(&["P"], &[("city", s("nyc")), ("tier", n(1.0))]);
        b.node(&["P"], &[("city", s("nyc")), ("tier", n(2.0))]);
        b.node(&["P"], &[("city", s("nyc")), ("tier", n(1.0))]); // dup of row 0
        let store = b.build();
        let plan = scan("P")
            .project(vec![
                ("city".into(), prop(0, "city")),
                ("tier".into(), prop(0, "tier")),
            ])
            .distinct();
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 2);
        assert_eq!(num(&out.rows[0][1]), 1.0);
        assert_eq!(num(&out.rows[1][1]), 2.0);
    }

    /// DISTINCT uses the grouping notion, not predicate equality: two NaNs
    /// collapse to one row.
    #[test]
    fn distinct_collapses_nans() {
        let mut b = Builder::default();
        b.node(&["P"], &[("v", n(f64::NAN))]);
        b.node(&["P"], &[("v", n(f64::NAN))]);
        b.node(&["P"], &[("v", n(1.0))]);
        let store = b.build();
        let plan = scan("P")
            .project(vec![("v".into(), prop(0, "v"))])
            .distinct();
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 2); // one NaN row + one 1.0 row
    }

    /// DISTINCT after Expand: the set of nodes reached by KNOWS from anyone.
    /// DISTINCT over a VAR-LENGTH endpoint keeps only the reachable-endpoint SET (the
    /// endpoint-dedup walk), byte-identical to materialize-then-dedup: `t` is reached by two
    /// length-2 paths (s→m1→t, s→m2→t) yet appears once.
    #[test]
    fn distinct_varlength_endpoint_set() {
        let mut b = Builder::default();
        let src = b.node(&["N"], &[("name", s("s")), ("score", n(1.0))]);
        let m1 = b.node(&["N"], &[("name", s("m1")), ("score", n(50.0))]);
        let m2 = b.node(&["N"], &[("name", s("m2")), ("score", n(99.0))]);
        let t = b.node(&["N"], &[("name", s("t")), ("score", n(5.0))]);
        b.edge(src, m1, "R");
        b.edge(src, m2, "R");
        b.edge(m1, t, "R");
        b.edge(m2, t, "R");
        let st = b.build();
        let sorted = |q: &str| {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
            v.sort();
            v
        };
        // {1,2}: len1 → m1,m2 ; len2 → t (via m1 and via m2, deduped). s is never reached.
        assert_eq!(
            sorted("MATCH (a)-[:R]->{1,2}(x) RETURN DISTINCT x.name AS n"),
            vec!["m1", "m2", "t"]
        );
        // {2,2}: exactly two hops → only t (once, despite two paths).
        assert_eq!(
            sorted("MATCH (a)-[:R]->{2,2}(x) RETURN DISTINCT x.name AS n"),
            vec!["t"]
        );
        // Endpoint WHERE (score < 60) applied over the deduped endpoints: m1(50), t(5) pass;
        // m2(99) fails. Reachable set {m1,m2,t} → {m1,t}.
        assert_eq!(
            sorted("MATCH (a)-[:R]->{1,2}(x) WHERE x.score < 60 RETURN DISTINCT x.name AS n"),
            vec!["m1", "t"]
        );
        // DISTINCT over an ENDPOINT EXPRESSION: upper(name) over {m1,m2,t} → {M1,M2,T}.
        // Exercises try_distinct_varlen_expr (dedup endpoints, then project the expression).
        assert_eq!(
            sorted("MATCH (a)-[:R]->{1,2}(x) RETURN DISTINCT upper(x.name) AS n"),
            vec!["M1", "M2", "T"]
        );
        // Var-length in the MIDDLE (a fixed R hop AFTER it): reachable x {m1,m2,t}, then
        // x→y: m1→t, m2→t, t→none. Distinct y set = {t}. Dedups at every hop.
        assert_eq!(
            sorted("MATCH (a)-[:R]->{1,2}(x)-[:R]->(y) RETURN DISTINCT y.name AS n"),
            vec!["t"]
        );
    }

    #[test]
    fn distinct_reached_set() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .project(vec![("who".into(), prop(1, "name"))])
            .distinct();
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["bob", "carol"]);
    }

    /// DISTINCT over a hop endpoint with DUPLICATE endpoints (a node reached by several
    /// edges) and a shared high-card value: the node-dedup fast path must skip the
    /// duplicate node yet still collapse two DIFFERENT nodes carrying the same string —
    /// single-column and composite (multi-column).
    #[test]
    fn distinct_frontier_dedups_duplicate_endpoints() {
        let mut bd = Builder::default();
        let b0 = bd.node(&["N"], &[("name", s("alpha")), ("city", s("x"))]);
        let b1 = bd.node(&["N"], &[("name", s("beta")), ("city", s("y"))]);
        let b2 = bd.node(&["N"], &[("name", s("alpha")), ("city", s("x"))]); // diff node, same values
        let a0 = bd.node(&["N"], &[]);
        let a1 = bd.node(&["N"], &[]);
        let a2 = bd.node(&["N"], &[]);
        let a3 = bd.node(&["N"], &[]);
        bd.edge(a0, b0, "R");
        bd.edge(a1, b0, "R"); // b0 reached twice → duplicate endpoint node
        bd.edge(a2, b1, "R");
        bd.edge(a3, b2, "R"); // same (name, city) via a different node
        let st = bd.build();
        // Single-column (Str frontier path): distinct names collapse b0's duplicate AND
        // b2's shared name → {alpha, beta}.
        let mut got = names_of(
            &run(
                &crate::gql::parse("MATCH (a)-[:R]->(x) RETURN DISTINCT x.name AS n").unwrap(),
                &st,
            ),
            0,
        );
        got.sort();
        assert_eq!(got, vec!["alpha", "beta"]);
        // Composite (multi-column frontier path): (alpha,x) appears via b0 and b2 → one row.
        assert_eq!(
            run(
                &crate::gql::parse("MATCH (a)-[:R]->(x) RETURN DISTINCT x.name AS n, x.city AS c")
                    .unwrap(),
                &st,
            )
            .rows
            .len(),
            2
        );
    }

    // --- Join (multi-pattern / shared variable) ---

    /// `MATCH (a)-[:KNOWS]->(b), (a)-[:WORKS_ON]->(c)` sharing `a`. Left slots
    /// [a,b], right slots [a,c]; join on left a (0) == right a (0); output slots
    /// [a, b, a', c]. Only alice has a WORKS_ON, so only her KNOWS rows survive.
    #[test]
    fn join_shared_start_variable() {
        let store = social();
        let left = scan("Person").expand(0, Dir::Out, &["KNOWS".to_string()]);
        let right = scan("Person").expand(0, Dir::Out, &["WORKS_ON".to_string()]);
        let plan = Plan::join(left, right, vec![(0, 0)]).project(vec![
            ("a".into(), prop(0, "name")),
            ("b".into(), prop(1, "name")),
            ("c".into(), prop(3, "name")), // right slot 1 -> output slot 2+1=3
        ]);
        let out = run(&plan, &store);
        let mut pairs: Vec<(String, String, String)> = out
            .rows
            .iter()
            .map(|r| (as_str(&r[0]), as_str(&r[1]), as_str(&r[2])))
            .collect();
        pairs.sort();
        // alice KNOWS {bob,carol}, WORKS_ON {graphdb}: 2x1 = 2 rows. bob has no
        // WORKS_ON, so bob->carol drops.
        assert_eq!(
            pairs,
            vec![
                ("alice".into(), "bob".into(), "graphdb".into()),
                ("alice".into(), "carol".into(), "graphdb".into()),
            ]
        );
    }

    /// The join fans out to the PRODUCT per shared key: a person with 2 R and 2 S
    /// neighbours yields 4 combined rows.
    #[test]
    fn join_is_product_per_shared_key() {
        let mut b = Builder::default();
        let a = b.node(&["P"], &[("name", s("a"))]);
        let r1 = b.node(&["P"], &[("name", s("r1"))]);
        let r2 = b.node(&["P"], &[("name", s("r2"))]);
        let s1 = b.node(&["P"], &[("name", s("s1"))]);
        let s2 = b.node(&["P"], &[("name", s("s2"))]);
        b.edge(a, r1, "R");
        b.edge(a, r2, "R");
        b.edge(a, s1, "S");
        b.edge(a, s2, "S");
        let store = b.build();
        let left = scan("P").expand(0, Dir::Out, &["R".to_string()]);
        let right = scan("P").expand(0, Dir::Out, &["S".to_string()]);
        let plan = Plan::join(left, right, vec![(0, 0)]).project(vec![
            ("r".into(), prop(1, "name")),
            ("s".into(), prop(3, "name")),
        ]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 4); // {r1,r2} x {s1,s2}
        let mut pairs: Vec<(String, String)> = out
            .rows
            .iter()
            .map(|r| (as_str(&r[0]), as_str(&r[1])))
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("r1".into(), "s1".into()),
                ("r1".into(), "s2".into()),
                ("r2".into(), "s1".into()),
                ("r2".into(), "s2".into()),
            ]
        );
    }

    /// A left key with no right match drops (inner join).
    #[test]
    fn join_drops_unmatched() {
        let store = social();
        // Everyone with a KNOWS edge, joined to everyone with a WORKS_ON edge on
        // the SAME person. Only alice has both, so bob (KNOWS only) drops.
        let left = scan("Person").expand(0, Dir::Out, &["KNOWS".to_string()]);
        let right = scan("Person").expand(0, Dir::Out, &["WORKS_ON".to_string()]);
        let plan = Plan::join(left, right, vec![(0, 0)])
            .project(vec![("a".into(), prop(0, "name"))])
            .distinct();
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["alice"]);
    }

    // --- VarLength (quantified hops) ---

    /// A linear chain a->b->c. `{1,2}` from a reaches b (len 1) and c (len 2):
    /// two rows.
    #[test]
    fn varlen_chain_one_to_two() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Trail)
            .project(vec![("end".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["b", "c"]);
    }

    /// `{0,2}` includes the source itself at length 0: a, b, c.
    #[test]
    fn varlen_zero_includes_source() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .var_length(0, Dir::Out, &["R".to_string()], 0, 2, PathMode::Trail)
            .project(vec![("end".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["a", "b", "c"]); // a at length 0
    }

    /// THE trail-vs-walk discriminator: a single self-loop a->a. `{1,2}`:
    /// - walk (trail=false) reuses the edge, so len1 AND len2 both reach a -> 2 rows;
    /// - trail (trail=true) may not reuse it, so only len1 -> 1 row.
    #[test]
    fn varlen_trail_vs_walk_on_a_self_loop() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        b.edge(a, a, "R"); // self-loop
        let store = b.build();
        let base = scan("N");

        let walk = base
            .clone()
            .var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Walk)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&walk, &store).rows.len(), 2, "walk reuses the edge");

        let trail = base
            .var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Trail)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(
            run(&trail, &store).rows.len(),
            1,
            "trail may not reuse the edge"
        );
    }

    /// A 2-cycle a<->b (two directed edges a->b, b->a). `{1,3}` as a TRAIL from a:
    /// len1 a->b (edge0); len2 a->b->a (edge0,edge1); len3 a->b->a->b (edge0,
    /// edge1, then edge0 again -> reused -> blocked). So endpoints b, a -> 2 rows.
    /// As a WALK, len3 a->b->a->b is allowed -> endpoints b, a, b -> 3 rows.
    #[test]
    fn varlen_two_cycle_trail_bounds_edge_reuse() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        b.edge(a, bb, "R"); // edge 0
        b.edge(bb, a, "R"); // edge 1
        let store = b.build();
        let from_a = scan("N").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))));

        let trail = from_a
            .clone()
            .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Trail)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&trail, &store).rows.len(), 2); // b (len1), a (len2)

        let walk = from_a
            .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Walk)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&walk, &store).rows.len(), 3); // b, a, b
    }

    /// Build the triangle a->b->c->a with a spur a->d — the ACYCLIC/SIMPLE fixture.
    #[cfg(test)]
    fn triangle_with_spur() -> Store {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        let d = b.node(&["N"], &[("name", s("d"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        b.edge(c, a, "R"); // closes the cycle back to the start
        b.edge(a, d, "R");
        b.build()
    }

    /// ACYCLIC forbids repeating ANY node — the hop c->a back to the start is
    /// rejected, so from a over `{1,3}` the endpoints are b, c, d (never a).
    #[test]
    fn varlen_acyclic_forbids_revisiting_the_start() {
        let store = triangle_with_spur();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Acyclic)
            .project(vec![("end".into(), prop(1, "name"))]);
        let mut got = names_of(&run(&plan, &store), 0);
        got.sort();
        assert_eq!(got, vec!["b", "c", "d"]); // no `a`: acyclic can't cycle back
    }

    /// SIMPLE forbids repeating an INTERIOR node but PERMITS a path that closes on
    /// its own start (start == end). From a over `{1,3}` the cycle a->b->c->a is a
    /// legal simple (closed) path, so `a` is emitted alongside b, c, d.
    #[test]
    fn varlen_simple_allows_the_closing_cycle() {
        let store = triangle_with_spur();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Simple)
            .project(vec![("end".into(), prop(1, "name"))]);
        let mut got = names_of(&run(&plan, &store), 0);
        got.sort();
        assert_eq!(got, vec!["a", "b", "c", "d"]); // `a` via the closing cycle
    }

    /// Over a 2-cycle a<->b from a with `{1,4}`, the count driver must respect the
    /// node modes (not the algebraic trail shortcut): SIMPLE emits b (len1) and the
    /// closing a (len2) = 2; ACYCLIC emits only b = 1 (a would repeat the start).
    #[test]
    fn varlen_count_honors_node_modes() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        b.edge(a, bb, "R");
        b.edge(bb, a, "R");
        let store = b.build();
        let from_a = scan("N").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))));
        let count = |mode| {
            let plan = from_a
                .clone()
                .var_length(0, Dir::Out, &["R".to_string()], 1, 4, mode)
                .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
            match run(&plan, &store).rows[0][0] {
                Value::Num(n) => n,
                ref other => panic!("want num, got {other:?}"),
            }
        };
        assert_eq!(count(PathMode::Simple), 2.0);
        assert_eq!(count(PathMode::Acyclic), 1.0);
    }

    /// Exact length `{2,2}` emits only the 2-hop endpoints.
    #[test]
    fn varlen_exact_length() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .var_length(0, Dir::Out, &["R".to_string()], 2, 2, PathMode::Trail)
            .project(vec![("end".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["c"]); // only the 2-hop endpoint
    }

    // --- ShortestPath ---

    /// A diamond a->b, a->c, b->d, c->d. Shortest from a: b(1), c(1), d(2). `d` is
    /// reachable two ways at distance 2 but emitted ONCE (ANY-shortest).
    #[test]
    fn shortest_path_diamond_reaches_each_once() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        let d = b.node(&["N"], &[("name", s("d"))]);
        b.edge(a, bb, "R");
        b.edge(a, c, "R");
        b.edge(bb, d, "R");
        b.edge(c, d, "R");
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .shortest_path(
                0,
                Dir::Out,
                &["R".to_string()],
                1,
                None,
                crate::ir::ShortestSelector::Any,
                None,
            )
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["b", "c", "d"]); // d once, not twice
    }

    /// The source is not emitted, and a direct edge wins over a longer path: with
    /// a->c direct AND a->b->c, c is reached at distance 1, once.
    #[test]
    fn shortest_path_takes_the_short_route() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        b.edge(a, c, "R"); // direct shortcut
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .shortest_path(
                0,
                Dir::Out,
                &["R".to_string()],
                1,
                None,
                crate::ir::ShortestSelector::Any,
                None,
            )
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["b", "c"]); // both at distance 1; source a not emitted
    }

    /// `max` caps the hop distance: on a chain a->b->c->d with max 2, d (distance
    /// 3) is unreachable.
    #[test]
    fn shortest_path_respects_max_hops() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        let d = b.node(&["N"], &[("name", s("d"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        b.edge(c, d, "R");
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .shortest_path(
                0,
                Dir::Out,
                &["R".to_string()],
                1,
                Some(2),
                crate::ir::ShortestSelector::Any,
                None,
            )
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["b", "c"]); // d (distance 3) beyond the cap
    }

    /// A cycle does not loop forever — each node is reached once. With a `+`
    /// (min 1) quantifier the source IS a valid endpoint at the shortest CYCLE
    /// length back to it (a->b->c->a is length 3), matching core.
    #[test]
    fn shortest_path_terminates_on_a_cycle() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        b.edge(c, a, "R"); // cycle back
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .shortest_path(
                0,
                Dir::Out,
                &["R".to_string()],
                1,
                None,
                crate::ir::ShortestSelector::Any,
                None,
            )
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        // b(1), c(2), and a(3) — the source closes the shortest cycle back to itself.
        assert_eq!(got, vec!["a", "b", "c"]);
    }

    // --- Lineage (path) ---

    /// A chain a->b->c. `RETURN path` over the 2-hop expand yields the hand-
    /// computed path [a, b, c] (node ids), and the path grows one node per hop.
    #[test]
    fn path_is_the_hop_sequence() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        let store = b.build();
        // (a)-[:R]->(x)-[:R]->(y) starting at a, RETURN path.
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .expand(0, Dir::Out, &["R".to_string()])
            .expand(1, Dir::Out, &["R".to_string()])
            .project(vec![("p".into(), Expr::Path)]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1);
        // path = [a, b, c] as node ids (a=0, b=1, c=2).
        match &out.rows[0][0] {
            Value::List(items) => {
                let ids: Vec<f64> = items
                    .iter()
                    .map(|v| match v {
                        Value::Num(x) => *x,
                        other => panic!("path element not a node id: {other:?}"),
                    })
                    .collect();
                assert_eq!(ids, vec![f64::from(a), f64::from(bb), f64::from(c)]);
            }
            other => panic!("expected a path list, got {other:?}"),
        }
    }

    /// Expand tracks the traversed EDGE in the lineage too: over a->b->c the
    /// relationships accessor recovers edge ids [0, 1] (creation order), the
    /// parallel of `path_is_the_hop_sequence` for edges.
    #[test]
    fn expand_lineage_tracks_edges() {
        use crate::ir::PathPart;
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R"); // edge id 0
        b.edge(bb, c, "R"); // edge id 1
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .expand(0, Dir::Out, &["R".to_string()])
            .expand(1, Dir::Out, &["R".to_string()])
            .project(vec![(
                "es".into(),
                Expr::PathAccess {
                    part: PathPart::Relationships,
                },
            )]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1);
        match &out.rows[0][0] {
            Value::List(items) => {
                let eids: Vec<f64> = items
                    .iter()
                    .map(|v| match v {
                        Value::Num(x) => *x,
                        other => panic!("edge element not an id: {other:?}"),
                    })
                    .collect();
                assert_eq!(eids, vec![0.0, 1.0]);
            }
            other => panic!("expected an edge list, got {other:?}"),
        }
    }

    /// A one-hop path is two nodes; the source's own path (length-0 walk via a
    /// bare scan) is one node.
    #[test]
    fn path_length_grows_with_hops() {
        let store = social();
        // alice -KNOWS-> {bob, carol}; RETURN path per edge.
        let plan = scan("Person")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))))
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .project(vec![("p".into(), Expr::Path)]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 2); // alice->bob, alice->carol
        for row in &out.rows {
            match &row[0] {
                Value::List(items) => assert_eq!(items.len(), 2), // [alice, neighbour]
                other => panic!("expected path list, got {other:?}"),
            }
        }
    }

    /// GATING: a lineage-free plan builds NO sidecar (pays nothing). Only a plan
    /// that reads Path tracks it. Checked at the batch level via `needs_lineage`
    /// and the pulled batch's `lineage` field.
    #[test]
    fn lineage_free_plan_builds_no_sidecar() {
        let store = social();
        let plain = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .project(vec![("b".into(), prop(1, "name"))]);
        assert!(!super::needs_lineage(&plain), "no Path read -> no lineage");
        // The pulled batch (before the lineage-dropping Project) has no sidecar.
        let inner = scan("Person").expand(0, Dir::Out, &["KNOWS".to_string()]);
        assert!(super::pull(&inner, &store, false)
            .unwrap()
            .lineage
            .is_none());

        let with_path = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .project(vec![("p".into(), Expr::Path)]);
        assert!(super::needs_lineage(&with_path), "Path read -> lineage");
        // With track=true the expand carries a sidecar.
        assert!(super::pull(&inner, &store, true).unwrap().lineage.is_some());
    }

    /// Lineage survives a reorder: ORDER BY over a path-tracking plan keeps each
    /// row's path aligned with its row.
    #[test]
    fn lineage_follows_a_reorder() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a")), ("age", n(1.0))]);
        let bb = b.node(&["N"], &[("name", s("b")), ("age", n(3.0))]);
        let c = b.node(&["N"], &[("name", s("c")), ("age", n(2.0))]);
        b.edge(a, bb, "R");
        b.edge(a, c, "R");
        let store = b.build();
        // a -> {b(age3), c(age2)}; order by the neighbour's age asc, RETURN path.
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .expand(0, Dir::Out, &["R".to_string()])
            .order_page(vec![asc(1, "age")], None, None)
            .project(vec![
                ("last".into(), prop(1, "name")),
                ("p".into(), Expr::Path),
            ]);
        let out = run(&plan, &store);
        // sorted by neighbour age: c(2) then b(3). Each path ends at its own node.
        assert_eq!(as_str(&out.rows[0][0]), "c");
        assert_eq!(as_str(&out.rows[1][0]), "b");
        let last_of = |row: &[Value]| match &row[1] {
            Value::List(items) => match items.last() {
                Some(Value::Num(x)) => *x,
                other => panic!("path tail not a node: {other:?}"),
            },
            other => panic!("expected path, got {other:?}"),
        };
        assert_eq!(last_of(&out.rows[0]), f64::from(c)); // path for c ends at c
        assert_eq!(last_of(&out.rows[1]), f64::from(bb)); // path for b ends at b
    }
}

#[cfg(test)]
mod perf {
    use crate::opt::optimize;
    use crate::store::{Builder, Store};
    use crate::value::Value;
    use std::sync::Arc;
    use std::time::Instant;

    fn build(nodes: usize, deg: usize) -> Store {
        let mut b = Builder::default();
        for i in 0..nodes {
            b.node(
                &["Person"],
                &[
                    ("name", Value::Str(Arc::from(format!("n{i}").as_str()))),
                    ("age", Value::Num((i % 100) as f64)),
                ],
            );
        }
        for i in 0..nodes {
            for d in 0..deg {
                b.edge(i as u32, ((i * 7 + d * 13 + 1) % nodes) as u32, "R");
            }
        }
        b.build()
    }

    #[test]
    #[ignore = "perf probe"]
    fn zzz_perf() {
        let (nodes, deg) = (200_000usize, 4usize);
        let t = Instant::now();
        let store = build(nodes, deg);
        eprintln!(
            "PERF build {nodes} nodes / {} edges: {:?}",
            nodes * deg,
            t.elapsed()
        );
        for q in [
            "MATCH (p:Person) WHERE p.age > 90 RETURN p.name",
            "MATCH (a:Person)-[:R]->(b) RETURN count(*) AS c",
            "MATCH (a:Person)-[:R]->(b) RETURN b.name AS who, count(*) AS c",
            "MATCH (a:Person)-[:R]->(b) RETURN b.age AS age, count(*) AS c",
            "MATCH (a:Person)-[:R]->()-[:R]->(c) RETURN count(DISTINCT c) AS c",
            "MATCH (a:Person)-[:R]->(b)-[:R]->(c) RETURN count(*) AS c",
        ] {
            let plan = optimize(super::super::gql::parse(q).unwrap());
            let mut best = f64::MAX;
            let mut rows = 0;
            for _ in 0..5 {
                let t = Instant::now();
                let out = super::run(&plan, &store);
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                rows = out.rows.len();
            }
            eprintln!("PERF {best:>9.2} ms  rows {rows:>8}  {q}");
        }
    }
}
