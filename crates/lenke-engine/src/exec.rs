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
        | Plan::MergeEdge { .. }
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
/// Resolve any UNBOXED element ref (`Value::Node`/`Value::Edge`, from a heterogeneous
/// branch/inject) to its element map at EGRESS — including refs nested inside a list/map/record
/// (`inject(v).fold()` → a list of maps). A plain scalar passes through untouched.
fn resolve_elem(v: Value, store: &Store) -> Value {
    match v {
        Value::Node(id) => node_result_value(store, id),
        Value::Edge(id) => edge_result_value(store, id),
        Value::List(xs) => Value::List(xs.into_iter().map(|x| resolve_elem(x, store)).collect()),
        Value::Map(pairs) => Value::Map(std::sync::Arc::new(
            pairs
                .iter()
                .map(|(k, val)| {
                    (
                        resolve_elem(k.clone(), store),
                        resolve_elem(val.clone(), store),
                    )
                })
                .collect(),
        )),
        Value::Record(fs) => Value::Record(
            fs.iter()
                .map(|(k, val)| (k.clone(), resolve_elem(val.clone(), store)))
                .collect(),
        ),
        other => other,
    }
}

fn col_into_values(col: Col, store: &Store) -> Vec<Value> {
    match col {
        Col::Str(data) => data.into_iter().map(Value::Str).collect(),
        Col::Num(data) => data.into_iter().map(Value::Num).collect(),
        Col::Bool(data) => data.into_iter().map(Value::Bool).collect(),
        Col::Gen(data) => data.into_iter().map(|v| resolve_elem(v, store)).collect(),
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
                out[i * ncols + c] = resolve_elem(v, store);
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
        Plan::MergeEdge {
            start_label,
            start_props,
            end_label,
            end_props,
            dir,
            etype,
            edge_props,
            on_create,
            on_update,
        } => execute_merge_edge(
            store,
            start_label,
            start_props,
            end_label,
            end_props,
            *dir,
            etype,
            edge_props,
            on_create,
            on_update,
        ),
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
    AddLabel(u32, String),
    RemoveLabel(u32, String),
    DeleteNode(u32, bool), // (node, detach)
    DeleteEdge(u32),       // eid
    SetEdge(u32, String, Value),
    RemoveEdge(u32, String),
}

mod write;
use self::write::*;

mod ddl;
use self::ddl::*;
// Re-exported at exec scope so existing `crate::exec::…` call sites resolve: ffi/lib
// reach apply_schema_op; binary.rs declares validators/invariants on binary load.
pub use self::ddl::apply_schema_op;
pub(crate) use self::ddl::{declare_invariant, declare_validator};

mod render;
use self::render::*;

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
        | Plan::MergeEdge { .. }
        | Plan::AddEdge { .. }
        | Plan::TxControl { .. } => Batch::of(Vec::new()),
        Plan::PathRecord { input, value, tag } => {
            let mut batch = pull(input, store, track)?;
            // Append this step's frontier value to each row's Gremlin step-history. `value`
            // is `Slot(frontier)` for a node/edge (its dense id) or the projected scalar; the
            // `tag` records which, so `path()` renders a vertex, an edge, or the raw value.
            // A `Slot` beyond the runtime width (a branch collapsed the layout, so the parser
            // frontier slot is gone) is skipped rather than evaluated — matching that a
            // path-through-a-branch is not yet the full history.
            let in_range = !matches!(value, Expr::Slot(n) if *n >= batch.slots.len());
            if track && in_range {
                if let Some(lin) = batch.lineage.take() {
                    let col = eval(value, store, &batch)?;
                    let vals: Vec<Value> = (0..batch.rows()).map(|i| col.value_at(i)).collect();
                    batch.lineage = Some(lin.push_step(&vals, *tag));
                }
            }
            batch
        }
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
            // The frontier is EDGES, not nodes. When a full `path()`/`tree()` is read
            // (`track`), seed the step-history with the source edge — `E().path()` yields
            // `[e]` per edge — so `PathRecord` can extend it with later steps.
            let ids = store.all_edges();
            let mut batch = Batch::single(Col::Edges(ids.clone()));
            if track {
                batch.lineage = Some(Lineage::seed_edges(&ids));
            }
            batch
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
            other,
        } => {
            // Edge frontier → endpoint vertex. Out=src (outV), In=dst (inV), Both=both
            // (fans out to two rows/edge). `other`=otherV: the endpoint the traverser did NOT
            // arrive from, read from the lineage's reference vertex. The endpoint lands in a
            // new appended slot; every other slot is carried through (duplicated for Both).
            let b = pull(input, store, track)?;
            let n = b.rows();
            let mut keep: Vec<usize> = Vec::new();
            let mut nodes: Vec<u32> = Vec::new();
            for i in 0..n {
                let eid = match b.slot(*edge_slot).value_at(i) {
                    Value::Num(x) if x >= 0.0 => x as u32,
                    // A branch/mixed frontier carries edges UNBOXED; a non-edge cell (a
                    // vertex/scalar from another arm) has no endpoint, so skip it.
                    Value::Edge(e) => e,
                    _ => continue,
                };
                let Some((src, dst)) = store.edge_endpoints(eid) else {
                    continue;
                };
                if *other {
                    // The reference vertex is the last node in the row's path (where the
                    // traverser arrived before this edge); otherV is the opposite endpoint. With
                    // no reference (an edge reached without a prior vertex, via a branch) default
                    // to the OUT vertex (src), matching pure-TS. A DIRECT bare edge source
                    // (`g.E().otherV()`) is rejected earlier, at parse.
                    let reference = b
                        .lineage
                        .as_ref()
                        .and_then(|l| otherv_reference(l.path_at(i), src, dst));
                    keep.push(i);
                    // Arrived from dst -> otherV is src; arrived from src (or NO reference: a
                    // bare edge reached via a branch) -> otherV is dst's opposite, i.e. src is
                    // the default OUT vertex, matching pure-TS.
                    nodes.push(match reference {
                        Some(r) if r == dst => src,
                        Some(_) => dst,
                        None => src,
                    });
                    continue;
                }
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
            // Read the node-id lineage DIRECTLY (not via `eval(Expr::Path)`, whose GQL
            // value is now a rich Path object, not a bare id list).
            let mut tree = GremlinTree::default();
            if let Some(lin) = &b.lineage {
                for i in 0..b.rows() {
                    let ids = lin.path_at(i);
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
            // A raw `order()` over a (possibly-element) frontier feeding the count carries a
            // runtime type-check the fast paths would elide — bail to general exec so it runs.
            let shortcuts_ok = !plan_has_raw_element_order(input);
            let mut out = if let Some(b) = shortcuts_ok
                .then(|| {
                    try_scan_count(input, keys, aggs, store)
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
                })
                .flatten()
            {
                b
            } else if let Some(b) = shortcuts_ok
                .then(|| try_frontier_group_fold(input, keys, aggs, store))
                .flatten()
            {
                b
            } else if let Some(b) = if shortcuts_ok {
                try_frontier_aggregate(input, keys, aggs, store)?
            } else {
                None
            } {
                b
            } else {
                aggregate(&pull(input, store, track)?, store, keys, aggs)?
            };
            // A GLOBAL reducer (`count()`/`sum()`/`fold()`/… — no group keys, one agg) collapses
            // the stream to a single value; TinkerPop RESETS the traverser path to [that value],
            // so a following `path()` reads it (`count().path()` → [7], and `count().path().path()`
            // nests [7, [7]]). Seed a fresh per-row step-history from the agg column when a path()
            // is read (`track`) — the fast paths above return no lineage. The node/edge path stays
            // empty (a reduced value is not a graph element).
            if track && keys.is_empty() && aggs.len() == 1 && !out.slots.is_empty() {
                let col = out.slot(out.slots.len() - 1);
                let vals: Vec<Value> = (0..out.rows()).map(|i| col.value_at(i)).collect();
                out.lineage = Some(crate::batch::Lineage::seed_steps(
                    &vals,
                    crate::batch::STEP_SCALAR,
                ));
            }
            out
        }
        Plan::OrderPage {
            input,
            keys,
            skip,
            limit,
            ..
        } if *limit == Some(0) => {
            // LIMIT 0 keeps no rows, so the input's projection is never evaluated —
            // a faulting expression (`RETURN 1/0 AS x LIMIT 0`) must yield the empty
            // result, not the fault. Short-circuit without pulling the input, but keep the
            // input's WIDTH (0 rows, N empty slots): a mid-chain `out().range(1,1).out()`
            // has a following step read a later slot, which a width-1 batch would panic on.
            let _ = (keys, skip);
            let w = crate::opt::width(input).max(1);
            Batch::of((0..w).map(|_| Col::Nodes(vec![])).collect())
        }
        Plan::OrderPage {
            input,
            keys,
            skip,
            limit,
            fault_on_element,
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
                order_page(&b, store, keys, *skip, *limit, *fault_on_element)?
            } else if let Some(b) = try_scan_top_k(input, keys, *skip, *limit, store, track) {
                // Streaming bounded top-K over a bare scan — no full frontier/idx array.
                b
            } else if let Some(b) = try_late_materialize(input, keys, *skip, *limit, store, track)?
            {
                // Sorted top-K over a projection: project only the surviving rows.
                b
            } else {
                order_page(
                    &pull(input, store, track)?,
                    store,
                    keys,
                    *skip,
                    *limit,
                    *fault_on_element,
                )?
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
            concat_batches(&subs, store)
        }
        Plan::PerElementBranch {
            input,
            kind,
            cond,
            arms,
            source_slot,
        } => {
            // TinkerPop coalesce/optional/choose are PER-TRAVERSER: each arm runs on ONE
            // incoming element, so a barrier/reducer inside an arm reduces that element's
            // sub-stream, not the whole batch. Run each input row through the arms on a
            // 1-row sub-batch and concatenate the per-row outputs in row order (the
            // interleave TinkerPop/the TS engine produce).
            let inb = pull(input, store, track)?;
            // An EMPTY frontier: per element there are NO elements to route, so the result is
            // empty — but `concat_batches(&[])` is a 0-slot batch and a downstream slot read
            // would index a column that isn't there. Reproduce the arms' natural width-1 shape
            // (typed column) and then take ZERO rows: running the arms whole-stream over the
            // empty input yields the right column TYPE, and `gather(&[])` drops any row a
            // reducer fabricated (`count()` over empty = [0]) so the output is truly empty.
            if inb.rows() == 0 {
                let subs: Vec<Batch> = arms
                    .iter()
                    .map(|b| pull_body(b, store, &inb))
                    .collect::<Result<_, _>>()?;
                return Ok(concat_batches(&subs, store).gather(&[]));
            }
            // The source element as a width-1 batch (the pass-through fallback), preserving
            // the row's lineage so a following path() still answers.
            let source_of = |sub: &Batch| -> Batch {
                let col = sub
                    .slots
                    .get(*source_slot)
                    .cloned()
                    .unwrap_or_else(|| Col::Gen(vec![Value::Null; sub.rows()]));
                let mut out = Batch::of(vec![col]);
                out.lineage = sub.lineage.clone();
                out
            };
            let mut outs: Vec<Batch> = Vec::with_capacity(inb.rows());
            for i in 0..inb.rows() {
                let sub = inb.gather(&[i]);
                let row_out = match kind {
                    crate::ir::PerElemKind::Coalesce => {
                        let mut chosen: Option<Batch> = None;
                        for arm in arms {
                            let r = pull_body(arm, store, &sub)?;
                            if r.rows() > 0 {
                                chosen = Some(r);
                                break;
                            }
                        }
                        match chosen {
                            Some(b) => b,
                            // No arm produced — an EMPTY row shaped like an arm's WIDTH-1 output
                            // (running arm[0] on the empty sub), NOT `sub.gather(&[])` which keeps
                            // the sub's full width: a leading hop makes sub width-2, and the width
                            // mismatch would desync the concat into a Gen column that a following
                            // fused hop mishandles (`out('KNOWS').coalesce(...).outE()` → null).
                            None => pull_body(&arms[0], store, &sub.gather(&[]))?,
                        }
                    }
                    crate::ir::PerElemKind::Optional => {
                        let r = pull_body(&arms[0], store, &sub)?;
                        if r.rows() > 0 {
                            r
                        } else {
                            source_of(&sub)
                        }
                    }
                    crate::ir::PerElemKind::Choose { has_else } => {
                        let c = pull_body(cond.as_ref().expect("choose has a cond"), store, &sub)?;
                        if c.rows() > 0 {
                            pull_body(&arms[0], store, &sub)?
                        } else if *has_else {
                            pull_body(&arms[1], store, &sub)?
                        } else {
                            source_of(&sub)
                        }
                    }
                };
                // A reducer arm (`count()`/`fold()`) collapses its sub-stream and drops the
                // lineage; TinkerPop resets the path to the reduced value, so seed a per-row
                // single-element step-history from the arm's output when a path() is read and the
                // arm produced none. Without it, `coalesce(values('name').count(), …).path()`
                // loses ALL lineage in the concat (all-or-nothing) and path() is [null].
                let mut row_out = row_out;
                if track && row_out.lineage.is_none() && !row_out.slots.is_empty() {
                    let vals: Vec<Value> = (0..row_out.rows())
                        .map(|i| row_out.slot(0).value_at(i))
                        .collect();
                    row_out.lineage = Some(crate::batch::Lineage::seed_steps(
                        &vals,
                        crate::batch::STEP_SCALAR,
                    ));
                }
                outs.push(row_out);
            }
            concat_batches(&outs, store)
        }
        Plan::Reconverge { input, slot } => {
            // Collapse to the single element/value column at `slot` (cloned, so a
            // Nodes/Edges frontier keeps its type), PRESERVING the lineage sidecar so a
            // reconverged branch arm still answers path(). A `slot` past the width (an
            // empty/zero-slice arm narrowed the batch) reads NULL.
            let b = pull(input, store, track)?;
            let col = b
                .slots
                .get(*slot)
                .cloned()
                .unwrap_or_else(|| Col::Gen(vec![Value::Null; b.rows()]));
            let mut out = Batch::of(vec![col]);
            out.lineage = b.lineage;
            out
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
            // A Project is row-preserving, so the path lineage stays aligned and flows
            // through — a following `PathRecord` (Gremlin value-step) needs it, and it does
            // not disturb the GQL node/edge path either (same rows).
            let mut out = Batch::of(cols);
            out.lineage = batch.lineage;
            out
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
            // Fast path: UNION ALL of same-width arms concatenates COLUMN-wise — but ONLY
            // when each column's arm variants agree. A mixed column (e.g. `V().inject(0)`:
            // a Nodes column unioned with a Num) would otherwise fall into `concat_cols`'
            // Gen fallback, which reads a node through `value_at` as its DENSE ID — losing
            // node identity, so a downstream `values('name')` sees a number and yields
            // nothing. The general path below renders such a node as its element map
            // (`render_cell`), matching the TS engine's heterogeneous stream.
            // UNION ALL preserves row order (no dedup), so a `path()` over an `inject`-mixed
            // stream survives: concatenate each arm's step-history, seeding a per-row single
            // element for an arm that has none (the injected literals' path is `[value]`).
            let arm_lin = |b: &Batch| -> crate::batch::Lineage {
                b.lineage.clone().unwrap_or_else(|| {
                    let vals: Vec<Value> = (0..b.rows()).map(|i| b.slot(0).value_at(i)).collect();
                    crate::batch::Lineage::seed_steps(&vals, crate::batch::STEP_SCALAR)
                })
            };
            let union_all_lineage = (matches!(op, CombineOp::Union) && *all && track)
                .then(|| crate::batch::Lineage::concat(&[&arm_lin(&bl), &arm_lin(&br)]));
            let variants_agree = (0..ncols).all(|j| same_col_variant(bl.slot(j), br.slot(j)));
            if matches!(op, CombineOp::Union) && *all && br.slots.len() == ncols && variants_agree {
                let mut out = concat_batches(&[bl, br], store);
                if let Some(l) = union_all_lineage {
                    out.lineage = Some(l);
                }
                return Ok(out);
            }
            let row_of = |b: &Batch, i: usize| -> Vec<Value> {
                let mut row: Vec<Value> = b.slots.iter().map(|c| cell_value(c, i, store)).collect();
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
            let mut out = Batch::of(cols.into_iter().map(Col::Gen).collect());
            // Same UNION ALL step-history concat as the fast path above (computed before the
            // arms were consumed).
            if let Some(l) = union_all_lineage {
                out.lineage = Some(l);
            }
            out
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
            // Nothing to dedup on an empty batch — and reading `key_slots` would panic
            // when a preceding empty/zero-slice hop narrowed the batch below the tagged
            // slot (`inE('UNKNOWN').otherV().range(0,0).dedup()` keeps the endpoint slot
            // in the plan's slot map but not in the 0-row batch's width).
            if batch.rows() == 0 {
                return Ok(batch);
            }
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
            optional,
            parts,
        } => {
            // Inline correlated (lateral) subquery: run `body` over the outer rows
            // (it is rooted at `Plan::Row`, which yields them), then emit one row
            // per sub-row — the outer slots the sub-row still carries, followed by
            // the yield expressions. Outer rows with no sub-row drop out (inner
            // lateral join), UNLESS `optional`.
            //
            // Slot `ow` seeds a PROVENANCE id (the outer row index) the body carries
            // through; OPTIONAL reads it to find which outer rows produced no sub-row.
            // The body's own variables land at `ow + 1…`.
            let outer = pull(input, store, track)?;
            let ow = *outer_width;
            let n = outer.rows();
            let mut seed_slots = outer.slots.clone();
            seed_slots.push(Col::Num((0..n).map(|i| i as f64).collect()));
            let seed = Batch::of(seed_slots);
            if !parts.is_empty() {
                return call_inline_setop(
                    store, &outer, &seed, ow, n, body, yields, parts, *optional,
                );
            }
            let sub = pull_body(body, store, &seed)?;
            let mut out_slots: Vec<Col> = (0..ow).map(|i| sub.slot(i).clone()).collect();
            for (_, e) in yields {
                out_slots.push(eval(e, store, &sub)?);
            }
            let mut matched = Batch::of(out_slots);
            // Carry any path the sub-rows accumulated (present only under lineage).
            matched.lineage = sub.lineage.clone();
            if *optional {
                // LEFT-outer: an outer row that produced no sub-row survives once,
                // with every yield column NULL. The prov column (slot `ow` of `sub`)
                // records which outer row each sub-row came from.
                let prov = sub.slot(ow);
                let mut seen = vec![false; n];
                for r in 0..sub.rows() {
                    if let Value::Num(p) = prov.value_at(r) {
                        let i = p as usize;
                        if i < n {
                            seen[i] = true;
                        }
                    }
                }
                let missing: Vec<usize> = (0..n).filter(|&i| !seen[i]).collect();
                if !missing.is_empty() {
                    // Build a fill SEED shaped like the body's row: the imported
                    // scope vars (slots < ow) keep their outer value; every body slot
                    // (prov and the subquery's own variables, >= ow) is NULL. Evaluate
                    // the yields over it — so a `RETURN *` that re-yields an imported
                    // var keeps it, while a fresh body var yields NULL (ISO: the outer
                    // row survives with the imported binding intact, the new one unbound).
                    let sub_width = sub.slots.len();
                    let k = missing.len();
                    let fill_seed = Batch::of(
                        (0..sub_width)
                            .map(|j| {
                                if j < ow {
                                    outer.slot(j).gather(&missing)
                                } else {
                                    // NULL of the body slot's own variant, so a node/edge
                                    // yield stays a node/edge column (u32::MAX sentinel → NULL)
                                    // rather than downgrading to Gen — which keeps `f.name`
                                    // (property access on a node column) resolving to NULL.
                                    match sub.slot(j) {
                                        Col::Nodes(_) => Col::Nodes(vec![u32::MAX; k]),
                                        Col::Edges(_) => Col::Edges(vec![u32::MAX; k]),
                                        _ => Col::Gen(vec![Value::Null; k]),
                                    }
                                }
                            })
                            .collect(),
                    );
                    let mut fill_slots: Vec<Col> =
                        (0..ow).map(|j| fill_seed.slot(j).clone()).collect();
                    for (_, e) in yields {
                        fill_slots.push(eval(e, store, &fill_seed)?);
                    }
                    let fill = Batch::of(fill_slots);
                    // The null-fill rows carry no path; drop lineage rather than
                    // desync it (OPTIONAL CALL over a path binding is niche).
                    matched.lineage = None;
                    matched = concat_batches(&[matched, fill], store);
                }
            }
            matched
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
            other,
        } => {
            // otherV needs the lineage reference vertex; the streaming fast path is
            // lineage-free, so it cannot take it.
            if *other {
                return None;
            }
            let (body, ids) = streaming_chain(input, store)?;
            Some((
                Plan::EdgeVertex {
                    input: Box::new(body),
                    edge_slot: *edge_slot,
                    which: *which,
                    other: false,
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
    Ok(Some(concat_batches(&acc, store)))
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
    Ok(Some(concat_batches(&acc, store)))
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
    Ok(Some(concat_batches(&acc, store)))
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

mod json_out;
pub use self::json_out::{
    run_gremlin_json, try_run_gql_json, try_run_gremlin_json, try_stream_gremlin_json,
};

mod order;
use self::order::*;

mod aggregation;
use self::aggregation::*;

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
    // A node frontier expands directly (the hot path, borrowed). A heterogeneous `Col::Gen`
    // (a mixed branch / inject) expands only its UNBOXED `Value::Node` cells — a scalar or an
    // edge cell contributes no vertex-neighbours (matches the pure-TS heterogeneous stream);
    // `u32::MAX` marks a skip. Any other column type has no node to expand.
    let src_owned: Vec<u32>;
    let src: &[u32] = match batch.slot(from) {
        Col::Nodes(v) => v,
        Col::Gen(cells) => {
            src_owned = cells
                .iter()
                .map(|c| match c {
                    Value::Node(id) => *id,
                    _ => u32::MAX,
                })
                .collect();
            &src_owned
        }
        _ => return empty(),
    };

    // Collect edge ids only when something needs them — a bound edge slot or
    // lineage — so the lineage-free hot path pushes nothing extra per neighbour.
    let track = batch.lineage.is_some();
    let need_eids = bind_edge || track;
    let mut keep = Vec::new();
    let mut nbrs = Vec::new();
    let mut eids = Vec::new();
    for (row, &v) in src.iter().enumerate() {
        if v == u32::MAX {
            continue; // an optional-null or a non-node Gen cell — no neighbours
        }
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
/// A raw `order()` / `order().by(desc)` — a bare self-slot sort key — somewhere on the linear
/// spine feeding a count. It sorts the current element itself, which faults over graph elements
/// at RUNTIME (see [`order_page`]); a build-time fault only fires when the frontier is a KNOWN
/// element, and a post-branch frontier is unknown, so this shape reaches the runtime check.
/// The count fast-paths peel `order()` away (a sort cannot change a count), which would ALSO
/// swallow that fault and silently return a count where pure-TS rejects the query. When one is
/// present, bail to general execution so the order runs and rejects an element frontier. A keyed
/// `by('k')`/`by(id)` sort projects a comparable scalar and is fine — its key is not a bare
/// `Slot`, so it is not flagged here.
fn plan_has_raw_element_order(p: &Plan) -> bool {
    match p {
        Plan::OrderPage { input, keys, .. } => {
            keys.iter().any(|k| matches!(k.expr, Expr::Slot(_)))
                || plan_has_raw_element_order(input)
        }
        // The single-input wrappers the count fast-paths peel through — follow the spine.
        Plan::Expand { input, .. }
        | Plan::OptionalExpand { input, .. }
        | Plan::EdgeVertex { input, .. }
        | Plan::VarLength { input, .. }
        | Plan::Filter { input, .. }
        | Plan::Distinct { input }
        | Plan::DistinctBy { input, .. }
        | Plan::Sample { input, .. }
        | Plan::Enumerate { input, .. }
        | Plan::Tail { input, .. }
        | Plan::Project { input, .. }
        | Plan::Reconverge { input, .. }
        | Plan::SortLocal { input, .. }
        | Plan::PathRecord { input, .. }
        | Plan::NullPadIfEmpty { input, .. }
        | Plan::Branch { input, .. }
        | Plan::PerElementBranch { input, .. } => plan_has_raw_element_order(input),
        _ => false,
    }
}

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

mod fastpath;
use self::fastpath::*;

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
        let rows_plus1 = bufs.offsets.len();
        out.lineage = Some(Lineage {
            values: bufs.values,
            offsets: bufs.offsets,
            edges: bufs.edges,
            edge_offsets: bufs.edge_offsets,
            steps: Vec::new(),
            step_tag: Vec::new(),
            step_off: vec![0; rows_plus1],
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
        let rows_plus1 = path_offsets.len();
        out.lineage = Some(Lineage {
            values: path_values,
            offsets: path_offsets,
            edges: path_edges,
            edge_offsets: path_edge_offsets,
            steps: Vec::new(),
            step_tag: Vec::new(),
            step_off: vec![0; rows_plus1],
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
        let rows_plus1 = bufs.offsets.len();
        out.lineage = Some(Lineage {
            values: bufs.values,
            offsets: bufs.offsets,
            edges: bufs.edges,
            edge_offsets: bufs.edge_offsets,
            steps: Vec::new(),
            step_tag: Vec::new(),
            step_off: vec![0; rows_plus1],
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

mod evaluator;
use self::evaluator::*;

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
fn concat_batches(subs: &[Batch], store: &Store) -> Batch {
    if subs.is_empty() {
        return Batch::of(Vec::new());
    }
    // A branch/coalesce reconverges bodies that need not share a width — `out('X')`
    // (an expand, wider) beside `limit(3).label()` (a scalar projection, narrow). The
    // output is as WIDE as the widest arm; a narrower arm contributes NULL for the
    // columns it lacks, so a downstream slot read never indexes past a short arm (a
    // slot-out-of-bounds panic before this padding).
    let ncols = subs.iter().map(|b| b.slots.len()).max().unwrap_or(0);
    let cols: Vec<Col> = (0..ncols)
        .map(|j| {
            // NULL placeholders for the arms that have no column `j` — kept alive for
            // the borrow below; unused (empty) when every arm has the column.
            let fills: Vec<Col> = subs
                .iter()
                .map(|b| {
                    if j < b.slots.len() {
                        Col::Gen(Vec::new())
                    } else {
                        Col::Gen(vec![Value::Null; b.rows()])
                    }
                })
                .collect();
            let refs: Vec<&Col> = subs
                .iter()
                .enumerate()
                .map(|(k, b)| {
                    if j < b.slots.len() {
                        b.slot(j)
                    } else {
                        &fills[k]
                    }
                })
                .collect();
            concat_cols(&refs, store)
        })
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

/// Whether two columns hold the same `Col` variant — the guard for the Union
/// concat fast path. A `Nodes`/`Num` mismatch must NOT concat (see `Plan::Union`),
/// because the Gen fallback would surface a node as its dense id.
fn same_col_variant(a: &Col, b: &Col) -> bool {
    use std::mem::discriminant;
    discriminant(a) == discriminant(b)
}

/// Concatenate columns of (ideally) the same variant. Same variant → keep it and
/// extend the inner vector; MIXED variants → materialize every value into `Gen`,
/// rendering a node/edge as its ELEMENT MAP (`render_cell`) rather than its dense id
/// (`value_at`) — a heterogeneous union/branch arm (a Nodes column beside a Str one)
/// must still surface real vertices/edges downstream, not bare numbers.
fn concat_cols(cols: &[&Col], store: &Store) -> Col {
    let gen = || {
        Col::Gen(
            cols.iter()
                .flat_map(|c| (0..c.len()).map(|i| cell_value(c, i, store)))
                .collect(),
        )
    };
    macro_rules! same {
        ($variant:ident) => {{
            let mut v = Vec::new();
            for c in cols {
                if let Col::$variant(xs) = c {
                    v.extend(xs.iter().cloned());
                } else {
                    return gen();
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
        Some(Col::Gen(_)) => gen(),
    }
}

/// The per-outer-row set-op combine for an inline CALL body with a `UNION`/`EXCEPT`/
/// `INTERSECT` tail. Each arm is provenance-tagged; its yield tuples are grouped by the
/// outer row that produced them, then combined left-associatively PER GROUP with the same
/// multiset semantics as the top-level set operators. Output rows are laid out in outer
/// order: each outer row's combined tuples (or one NULL-yield row under OPTIONAL when its
/// group is empty). The outer columns are gathered natively (kept as Nodes/Edges cols) so
/// downstream `x.name` still resolves; the yield columns are materialized (`Gen`) like the
/// top-level set-op, which is what the dedup keys already compare.
#[allow(clippy::too_many_arguments)]
fn call_inline_setop(
    store: &Store,
    outer: &Batch,
    seed: &Batch,
    ow: usize,
    n: usize,
    body: &Plan,
    yields: &[(String, Expr)],
    parts: &[crate::ir::CallPart],
    optional: bool,
) -> Result<Batch, String> {
    let ny = yields.len();
    let key_of = |t: &[Value]| -> Vec<u8> {
        let mut buf = Vec::new();
        for v in t {
            value::group_key_into(v, &mut buf);
        }
        buf
    };
    // Collect one arm's yield tuples, grouped by provenance (the outer row index each
    // sub-row came from). A fresh-scan arm carries prov through the cross-join too.
    let collect =
        |arm_body: &Plan, arm_yields: &[(String, Expr)]| -> Result<Vec<Vec<Vec<Value>>>, String> {
            let sub = pull_body(arm_body, store, seed)?;
            let ycols: Vec<Col> = arm_yields
                .iter()
                .map(|(_, e)| eval(e, store, &sub))
                .collect::<Result<_, _>>()?;
            let prov = sub.slot(ow);
            let mut groups: Vec<Vec<Vec<Value>>> = vec![Vec::new(); n];
            for r in 0..sub.rows() {
                let p = match prov.value_at(r) {
                    Value::Num(x) => x as usize,
                    _ => continue,
                };
                if p < n {
                    groups[p].push(ycols.iter().map(|c| render_cell(c, r, store)).collect());
                }
            }
            Ok(groups)
        };
    let mut acc = collect(body, yields)?;
    for part in parts {
        let rhs = collect(&part.body, &part.yields)?;
        for (p, rgroup) in rhs.into_iter().enumerate() {
            let lgroup = std::mem::take(&mut acc[p]);
            acc[p] = combine_call_groups(part.op, part.all, lgroup, rgroup, &key_of);
        }
    }
    // Lay out output rows in outer order: each group's tuples, then a NULL-yield row
    // under OPTIONAL for an empty group.
    let mut provs: Vec<usize> = Vec::new();
    let mut ycols_out: Vec<Vec<Value>> = vec![Vec::new(); ny];
    for (p, group) in acc.iter().enumerate() {
        if group.is_empty() {
            if optional {
                provs.push(p);
                for col in ycols_out.iter_mut() {
                    col.push(Value::Null);
                }
            }
            continue;
        }
        for tuple in group {
            provs.push(p);
            for (k, v) in tuple.iter().enumerate() {
                ycols_out[k].push(v.clone());
            }
        }
    }
    let mut out_slots: Vec<Col> = (0..ow).map(|j| outer.slot(j).gather(&provs)).collect();
    for col in ycols_out {
        out_slots.push(Col::Gen(col));
    }
    Ok(Batch::of(out_slots))
}

/// Combine two provenance groups' yield tuples under one set operator — the multiset
/// rules of the top-level `Plan::Union` exec, applied within a single outer-row group.
fn combine_call_groups(
    op: crate::ir::CombineOp,
    all: bool,
    mut l: Vec<Vec<Value>>,
    r: Vec<Vec<Value>>,
    key_of: &impl Fn(&[Value]) -> Vec<u8>,
) -> Vec<Vec<Value>> {
    use crate::ir::CombineOp;
    match op {
        CombineOp::Union => {
            l.extend(r);
            if !all {
                let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
                l.retain(|t| seen.insert(key_of(t)));
            }
            l
        }
        CombineOp::Except | CombineOp::Intersect => {
            let mut rkeys: FnvSet<Vec<u8>> = FnvSet::default();
            for t in &r {
                rkeys.insert(key_of(t));
            }
            let want_present = matches!(op, CombineOp::Intersect);
            let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
            let mut out = Vec::new();
            for t in l {
                let k = key_of(&t);
                if rkeys.contains(&k) == want_present && seen.insert(k) {
                    out.push(t);
                }
            }
            out
        }
    }
}

fn pull_body(plan: &Plan, store: &Store, seed: &Batch) -> Result<Batch, String> {
    Ok(match plan {
        Plan::Row => seed.clone(),
        // A `path()`-through-branch records each arm step into the seed's step-history, so a
        // path() reads the arm's hops — the streaming twin of the main `PathRecord` arm.
        Plan::PathRecord { input, value, tag } => {
            let mut batch = pull_body(input, store, seed)?;
            let in_range = !matches!(value, Expr::Slot(n) if *n >= batch.slots.len());
            if in_range {
                if let Some(lin) = batch.lineage.take() {
                    let col = eval(value, store, &batch)?;
                    let vals: Vec<Value> = (0..batch.rows()).map(|i| col.value_at(i)).collect();
                    batch.lineage = Some(lin.push_step(&vals, *tag));
                }
            }
            batch
        }
        // A correlated fresh Scan (a CALL set-op arm that starts from `(x:Label)` rather
        // than a scope variable): cross-join every seed row with every matching node,
        // appending the node at the next slot. Ignores its position as a "leaf" — the
        // seed IS its input, so the prov column the seed carries fans out with it.
        Plan::Scan { label } => {
            let nodes: Vec<u32> = match label {
                Some(l) => store.nodes_with_label(l).to_vec(),
                None => (0..store.node_count() as u32).collect(),
            };
            let mut keep: Vec<usize> = Vec::with_capacity(seed.rows() * nodes.len());
            let mut ncol: Vec<u32> = Vec::with_capacity(seed.rows() * nodes.len());
            for r in 0..seed.rows() {
                for &nd in &nodes {
                    keep.push(r);
                    ncol.push(nd);
                }
            }
            let mut out = seed.gather(&keep);
            out.slots.push(Col::Nodes(ncol));
            out
        }
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
            other,
        } => {
            let b = pull_body(input, store, seed)?;
            let mut keep: Vec<usize> = Vec::new();
            let mut nodes: Vec<u32> = Vec::new();
            for i in 0..b.rows() {
                let eid = match b.slot(*edge_slot).value_at(i) {
                    Value::Num(x) if x >= 0.0 => x as u32,
                    // A branch/mixed frontier carries edges UNBOXED; a non-edge cell (a
                    // vertex/scalar from another arm) has no endpoint, so skip it.
                    Value::Edge(e) => e,
                    _ => continue,
                };
                let Some((src, dst)) = store.edge_endpoints(eid) else {
                    continue;
                };
                if *other {
                    let reference = b
                        .lineage
                        .as_ref()
                        .and_then(|l| otherv_reference(l.path_at(i), src, dst));
                    keep.push(i);
                    // Arrived from dst -> otherV is src; arrived from src (or NO reference: a
                    // bare edge reached via a branch) -> otherV is dst's opposite, i.e. src is
                    // the default OUT vertex, matching pure-TS.
                    nodes.push(match reference {
                        Some(r) if r == dst => src,
                        Some(_) => dst,
                        None => src,
                    });
                    continue;
                }
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
            let mut out = aggregate(&b, store, keys, aggs)?;
            // A GLOBAL reducer resets the traverser path (see the main Aggregate arm): seed a
            // fresh step-history [agg value] so a following path() reads the reduced value
            // (`union(count(), …).path()` → [count]).
            if seed.lineage.is_some() && keys.is_empty() && aggs.len() == 1 && !out.slots.is_empty()
            {
                let col = out.slot(out.slots.len() - 1);
                let vals: Vec<Value> = (0..out.rows()).map(|i| col.value_at(i)).collect();
                out.lineage = Some(crate::batch::Lineage::seed_steps(
                    &vals,
                    crate::batch::STEP_SCALAR,
                ));
            }
            out
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
            fault_on_element,
        } => {
            let b = pull_body(input, store, seed)?;
            order_page(&b, store, keys, *skip, *limit, *fault_on_element)?
        }
        Plan::Distinct { input } => distinct_batch(pull_body(input, store, seed)?),
        // Gremlin `dedup()` / `dedup('a',…)` inside a branch arm — first-seen per distinct
        // key tuple over the seeded body (same as the main path's materialized fallback).
        Plan::DistinctBy { input, key_slots } => {
            let batch = pull_body(input, store, seed)?;
            if batch.rows() == 0 {
                batch
            } else {
                let typed = distinct_by_typed(&batch, key_slots);
                let mut seen_ids: FnvSet<u32> = FnvSet::default();
                let mut seen_bytes: FnvSet<Vec<u8>> = FnvSet::default();
                let keep =
                    distinct_by_keep(&batch, key_slots, typed, &mut seen_ids, &mut seen_bytes);
                batch.gather(&keep)
            }
        }
        Plan::Reconverge { input, slot } => {
            // Collapse a branch arm to its element/value column at `slot` (cloned so a
            // Nodes/Edges frontier keeps its type), preserving the lineage sidecar so
            // path()-through-branch still works. `slot` past the width reads NULL.
            let b = pull_body(input, store, seed)?;
            let col = b
                .slots
                .get(*slot)
                .cloned()
                .unwrap_or_else(|| Col::Gen(vec![Value::Null; b.rows()]));
            let mut out = Batch::of(vec![col]);
            out.lineage = b.lineage;
            out
        }
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

mod scalar;
pub(crate) use self::scalar::temporal_ctor;
use self::scalar::*;

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

mod distinct;
use self::distinct::*;

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
        // A slot past the runtime width (a branch/inject collapsed the layout) has no column
        // to gate on — fall back to the general evaluator rather than indexing out of bounds.
        let Some(Col::Nodes(ids)) = batch.slots.get(*slot) else {
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
/// If `pairs` is a BOXED element map — a vertex `{id, labels, properties}` or an
/// edge `{id, from, to, labels, properties}`, the shape `render_cell` produces when
/// a heterogeneous union collapses a Nodes/Edges column into `Gen` — return its
/// nested `properties` map. A property read/existence check on such a value must
/// look INSIDE `properties` (`values('name')` on a boxed vertex is `props.name`,
/// not the top-level `name`, which does not exist). `None` for a plain map, whose
/// caller keeps the top-level-key behavior.
fn boxed_element_props(pairs: &[(Value, Value)]) -> Option<&std::sync::Arc<Vec<(Value, Value)>>> {
    let (mut has_id, mut has_labels) = (false, false);
    let mut props = None;
    for (k, v) in pairs {
        let Value::Str(ks) = k else {
            return None;
        };
        match ks.as_ref() {
            "id" => has_id = true,
            "labels" => has_labels = true,
            "from" | "to" => {}
            "properties" => match v {
                Value::Map(p) => props = Some(p),
                _ => return None,
            },
            _ => return None, // an unexpected key → a plain map, not an element
        }
    }
    // Exactly a vertex (3 keys) or an edge (5 keys) map.
    if has_id && has_labels && props.is_some() && (pairs.len() == 3 || pairs.len() == 5) {
        props
    } else {
        None
    }
}

/// The value of property `key` off a boxed `Value` — reading through a boxed element
/// map's `properties`, a record's field, or a plain map's top-level key.
fn boxed_value_prop(v: &Value, key: &str) -> Value {
    let lookup = |pairs: &[(Value, Value)]| {
        pairs
            .iter()
            .find(|(k, _)| matches!(k, Value::Str(s) if s.as_ref() == key))
            .map_or(Value::Null, |(_, v)| v.clone())
    };
    match v {
        Value::Record(fields) => value::record_field(fields, key),
        Value::Map(pairs) => match boxed_element_props(pairs) {
            Some(props) => lookup(props),
            None => lookup(pairs),
        },
        _ => Value::Null,
    }
}

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
        // field; an UNBOXED element ref (Value::Node/Edge in a heterogeneous Col::Gen)
        // reads its stored property off the store; anything else has no property → NULL.
        return Col::Gen(
            (0..col.len())
                .map(|i| match col.value_at(i) {
                    Value::Node(id) => store.prop(id, key),
                    Value::Edge(e) => store.edge_prop(e, key),
                    other => boxed_value_prop(&other, key),
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
pub(crate) const TRUTH_TYPE_ERR: &str =
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
mod tests;
