//! Evaluator + executor over the lowered IR ([`super::plan`]). Pattern matching
//! is a backtracking visitor over the columnar adjacency; expressions use ISO
//! three-valued (Kleene) logic. The IR has already resolved `$param` to a
//! positional slot, functions to enums, and projection metadata — so the
//! per-row path here is a plain `match`, no string work for params/functions.
//!
//! A query is run via a [`Prepared`] plan: lower once, execute many times with
//! different params (positional, slotted at lower time) against any graph.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicU8, Ordering as AtomOrdering};
use std::sync::Arc;

/// A fast, non-cryptographic hasher (FxHash — the one rustc uses) for the internal
/// grouping / dedup maps, where the default SipHash dominates: `GROUP BY <node>`
/// over a big result hashes a short key per row, and SipHash's ~40 ns there is the
/// wall. FxHash processes 8-byte words with a multiply-rotate-xor, ~3–4× faster on
/// these keys, and needs no dependency. These maps are internal (never keyed by
/// untrusted external data in a way that a hash-flood would matter), so the DoS
/// resistance SipHash buys is not needed here.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        // FxHash's raw accumulator has weak high-bit avalanche, so structured keys
        // (`@v0`, `@v1`, … — a common prefix + a small varying suffix) cluster and
        // the map probes more. A splitmix64 finalize (3 mul-xor-shift, once per
        // hash) fully mixes it — restoring good distribution while keeping the fast
        // per-word write. Without this, FxHash was *slower* than SipHash here.
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        let mut h = self.0;
        while bytes.len() >= 8 {
            let w = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            h = (h.rotate_left(5) ^ w).wrapping_mul(SEED);
            bytes = &bytes[8..];
        }
        if !bytes.is_empty() {
            let mut w = 0u64;
            for (i, &b) in bytes.iter().enumerate() {
                w |= (b as u64) << (i * 8);
            }
            h = (h.rotate_left(5) ^ w).wrapping_mul(SEED);
        }
        self.0 = h;
    }
    #[inline]
    fn write_u64(&mut self, w: u64) {
        const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(SEED);
    }
}

type FxBuild = BuildHasherDefault<FxHasher>;
type FxHashMap<K, V> = HashMap<K, V, FxBuild>;
type FxHashSet<K> = HashSet<K, FxBuild>;

#[cfg(feature = "parallel-query")]
use rayon::prelude::*;

use super::ast::{
    AccessMode, ArithOp, Clause, CompareOp, Direction, Lit, PathMode, PathSelector, Quantifier,
    Query, SetOp, SetOpKind, Statement, TxControl, TxKind, TypeTest,
};
use super::lexer::SyntaxError;
use super::plan::{
    has_argless_aggregate, has_nested_aggregate, lower, AggFn, CAgg, CClause, CCount, CElem, CExpr,
    CHop, CLabelExpr, CLinear, CMerge, CMergeUpdate, CNode, CPath, CPredicate, CProjection,
    CPropConstraint, CQuery, CRel, CRemoveItem, CReturnItem, CSegment, CSetItem, CUnit, Op,
    Program, ScalarFn,
};
#[cfg(feature = "arrow")]
use crate::arrow::ArrowColumn;
use crate::error::{CodeError, CodeResult};
use crate::error_codes::ErrorCode;
use crate::graph::{Adj, Column, Graph, TxCommitError, Value};
use crate::query::RowSet;

/// A runtime value.
///
/// An alias of the shared [`crate::value::Value`] — Gremlin's `GVal` is the same
/// type. The variants GQL uses are `Null`, `Bool`, `Num`, `Str`, `Temporal`,
/// `Node`, `Edge`, `List`, `Record` (the ISO record — Gremlin's insertion-ordered
/// `Map` is a different variant) and `Path`; `Map` and `Property` belong to
/// Gremlin and never appear here.
///
/// The type carries a `PartialEq`, and GQL MUST NOT use it: that one is
/// TinkerPop's. ISO equality is three-valued and lives in [`compare_vals`].
pub type Val = crate::value::Value;

/// One candidate solution: variable slot → value. Slots are assigned per scope
/// by the lowering pass, so access is an array index (not a name scan). `None` is
/// an unbound slot; `Some(Val::Null)` is an explicit null (e.g. OPTIONAL MATCH).
#[derive(Clone, Debug, Default)]
pub struct Binding(Vec<Option<Val>>);

impl Binding {
    /// An empty binding with room for `n` slots.
    fn with_len(n: usize) -> Self {
        Self(vec![None; n])
    }
    fn get(&self, slot: usize) -> Option<&Val> {
        self.0.get(slot).and_then(|o| o.as_ref())
    }
    fn bound(&self, slot: usize) -> bool {
        self.0.get(slot).is_some_and(|o| o.is_some())
    }
    fn set(&mut self, slot: usize, v: Val) {
        if slot >= self.0.len() {
            self.0.resize(slot + 1, None);
        }
        self.0[slot] = Some(v);
    }
    fn unset(&mut self, slot: usize) {
        if slot < self.0.len() {
            self.0[slot] = None;
        }
    }
    /// Move a slot's value OUT, leaving it unbound. For a consumer that owns the
    /// value from here on — the scan reading a quantified subpath's group-variable
    /// `Val::List` off the binding into its column — so the list is not deep-cloned
    /// only for the binding's copy to be dropped on the next line.
    fn take(&mut self, slot: usize) -> Option<Val> {
        self.0.get_mut(slot).and_then(Option::take)
    }
    fn resize(&mut self, len: usize) {
        self.0.resize(len, None);
    }
}

/// Query parameters supplied by name; bound to positional slots at execute time.
pub type Params = HashMap<String, Val>;

/// Per-execution context resolved once against the graph: positional params, and
/// each plan ref (property key / label name) resolved to its graph id so the
/// per-row path is an array index, not a `HashMap` lookup. It owns the resolved
/// tables and borrows nothing from the graph, so the write path can still take
/// `&mut Graph` alongside it.
/// Pool of reusable per-edge trail-mark buffers for var-length TRAIL walks
/// (`reachable_each`): pop a buffer, walk with it, push it back clean. Backed by a
/// `RefCell` in the default single-threaded build (a borrow flag — effectively free);
/// a `Mutex` only under `parallel-query`, where `Ctx` must be `Sync` to be shared
/// across rayon's projection threads. Only the pop/push is guarded — never the walk
/// itself — so even the parallel build contends for nothing more than the brief buffer
/// hand-off, and the default build pays no synchronization cost at all.
#[derive(Default)]
struct MarksPool {
    #[cfg(not(feature = "parallel-query"))]
    inner: std::cell::RefCell<Vec<Vec<bool>>>,
    #[cfg(feature = "parallel-query")]
    inner: std::sync::Mutex<Vec<Vec<bool>>>,
}

impl MarksPool {
    /// Take a buffer from the pool, or a fresh empty one if it's dry.
    fn pop(&self) -> Vec<bool> {
        #[cfg(not(feature = "parallel-query"))]
        let mut pool = self.inner.borrow_mut();
        #[cfg(feature = "parallel-query")]
        let mut pool = self.inner.lock().expect("edge-marks pool mutex poisoned");
        pool.pop().unwrap_or_default()
    }

    /// Return a buffer to the pool for reuse.
    fn push(&self, buf: Vec<bool>) {
        #[cfg(not(feature = "parallel-query"))]
        let mut pool = self.inner.borrow_mut();
        #[cfg(feature = "parallel-query")]
        let mut pool = self.inner.lock().expect("edge-marks pool mutex poisoned");
        pool.push(buf);
    }
}

struct Ctx<'a> {
    params: &'a [Val],
    /// key_ref -> (vertex property-key id, edge property-key id).
    prop_keys: Vec<(Option<u32>, Option<u32>)>,
    /// label ref -> (vertex-label id, edge-type id) — a name can be both.
    labels: Vec<(Option<u32>, Option<u32>)>,
    /// label ref -> name (for write clauses, which create labels/types by name).
    label_names: &'a [String],
    /// Unknown/unimplemented function names the plan references — named in the
    /// `UnknownFunction` error when one faults (see `FAULT_UNKNOWN_FN`).
    unknown_fns: &'a [String],
    /// First ISO data exception raised during evaluation (see `FAULT_*`). The
    /// infallible `eval`/VM/vectorized engines can't return `Err`, so they record
    /// the fault here and return a placeholder; the driver checks it at the row
    /// boundary and converts it to a `CodeError`. Atomic so the parallel (rayon)
    /// vectorized path can record faults safely.
    fault: AtomicU8,
    /// Pool of reusable per-edge trail-mark buffers for `reachable_each` (var-length
    /// TRAIL walk): `buf[eidx]` == "this edge is on the current trail." Pooled rather
    /// than `HashSet`-per-call to get O(1) index ops without re-allocating each call.
    /// A pool (not one buffer) because the walk is now *lazy*: it invokes a callback
    /// per endpoint that may re-enter `reachable_each` (a nested quantified segment)
    /// while the outer walk's marks are still live — so each active walk borrows its
    /// own buffer (`take_marks`) and returns it clean (`return_marks`). The pool lock is
    /// held only for the brief pop/push, never across the walk, so nesting is safe.
    edge_marks_pool: MarksPool,
}

const FAULT_NONE: u8 = 0;
const FAULT_DIV_ZERO: u8 = 1;
const FAULT_TYPE: u8 = 2;
const FAULT_BUDGET: u8 = 3;
const FAULT_BAD_LABEL: u8 = 4;
const FAULT_UNKNOWN_FN: u8 = 5;
const FAULT_CONSTRAINT: u8 = 6;
const FAULT_MERGE_KEY: u8 = 7;
const FAULT_MERGE_EDGE: u8 = 8;
const FAULT_REQUIRED: u8 = 9;
const FAULT_TYPE_CONSTRAINT: u8 = 10;
const FAULT_DURATION_OVERFLOW: u8 = 11;
const FAULT_DATE_OVERFLOW: u8 = 12;
const FAULT_TEMPORAL_AGG: u8 = 13;
const FAULT_NONNUMERIC_AGG: u8 = 14;
const FAULT_INTERMEDIATE: u8 = 15;
const FAULT_ID_DUP: u8 = 16;
const FAULT_ID_IMMUTABLE: u8 = 17;
const FAULT_CMP_TEMPORAL: u8 = 18;
const FAULT_DATE_PART: u8 = 19;
const FAULT_CARDINALITY: u8 = 20;
const FAULT_RANGE_BUDGET: u8 = 21;

impl Ctx<'_> {
    /// Re-resolve property-key and label ids against the current graph (keeping
    /// params/fault). Needed mid-INSERT: freshly created nodes introduce columns
    /// a snapshot taken before the clause doesn't know about, so a forward
    /// reference (`INSERT (a {..}), (:B {x: a.id})`) would otherwise read NULL.
    fn refresh_ids(&mut self, graph: &Graph, plan: &CQuery) {
        self.prop_keys = plan
            .key_names
            .iter()
            .map(|n| (graph.props.keys.get(n), graph.edge_props.keys.get(n)))
            .collect();
        self.labels = plan
            .label_names
            .iter()
            .map(|n| (graph.labels.get(n), graph.etype.get(n)))
            .collect();
    }

    /// Borrow an all-`false` trail-mark buffer sized for every edge slot, from the
    /// pool (or a fresh one). Returned clean by `return_marks`, so a pooled buffer is
    /// already all-`false`; only grow it if the graph gained edges since last use.
    fn take_marks(&self, slots: usize) -> Vec<bool> {
        let mut buf = self.edge_marks_pool.pop();
        if buf.len() < slots {
            buf.resize(slots, false);
        }
        buf
    }

    /// Return a trail-mark buffer to the pool. The caller must leave it all-`false`
    /// (backtracking clears it on the normal path; the stop/fault paths clear the
    /// live stack's marks before returning).
    fn return_marks(&self, buf: Vec<bool>) {
        self.edge_marks_pool.push(buf);
    }

    /// Record a data-exception fault (first one wins; later faults are ignored).
    #[inline]
    fn set_fault(&self, kind: u8) {
        if self.fault.load(AtomOrdering::Relaxed) == FAULT_NONE {
            self.fault.store(kind, AtomOrdering::Relaxed);
        }
    }

    /// Convert any recorded fault into an `Err`, to be called at a row boundary.
    fn check_fault(&self) -> CodeResult<()> {
        match self.fault.load(AtomOrdering::Relaxed) {
            FAULT_DIV_ZERO => Err(CodeError::new(ErrorCode::DataException, "division by zero")),
            FAULT_DURATION_OVERFLOW => Err(CodeError::new(
                ErrorCode::DataException,
                "duration overflow: a component exceeds the representable (float64-safe-integer) range",
            )),
            FAULT_DATE_OVERFLOW => Err(CodeError::new(
                ErrorCode::DataException,
                "date overflow: arithmetic result is outside the representable date range",
            )),
            FAULT_TEMPORAL_AGG => Err(CodeError::new(
                ErrorCode::DataException,
                "unsupported temporal aggregate: sum() is defined only for DURATION (dates/times \
                 aren't summable), and avg() over DURATION would need duration/count (often \
                 non-representable, e.g. avg(P1M,P2M)=P1.5M); use min()/max(), or sum() + host division",
            )),
            FAULT_NONNUMERIC_AGG => Err(CodeError::new(
                ErrorCode::DataException,
                "sum()/avg() require numeric values; a list/map is not summable — reduce it first \
                 (Gremlin sum(local), or GQL UNWIND + sum)",
            )),
            FAULT_TYPE => Err(CodeError::new(
                ErrorCode::DataException,
                "arithmetic requires a number",
            )),
            FAULT_CARDINALITY => Err(CodeError::new(
                ErrorCode::DataException,
                "a VALUE scalar subquery returned more than one row; add an aggregate \
                 (e.g. count/collect), a LIMIT-like bound, or a more selective pattern",
            )),
            FAULT_BUDGET => Err(CodeError::new(
                ErrorCode::ResourceExhausted,
                "variable-length pattern exceeded the trail budget; add a tighter bound",
            )),
            FAULT_RANGE_BUDGET => Err(CodeError::new(
                ErrorCode::ResourceExhausted,
                "range() would materialize more than 1000000 elements; narrow the bounds                  or widen the step",
            )),
            FAULT_INTERMEDIATE => Err(CodeError::new(
                ErrorCode::ResourceExhausted,
                "multi-hop pattern materialized too many intermediate rows; add selective \
                 per-hop predicates, anchor an endpoint, or shorten the chain",
            )),
            FAULT_ID_DUP => Err(CodeError::new(
                ErrorCode::ConstraintViolation,
                "an element with this id already exists — a string `id` property is the \
                 element's unique identity; use _MERGE to upsert, or a fresh id",
            )),
            FAULT_ID_IMMUTABLE => Err(CodeError::new(
                ErrorCode::InvalidGraphOp,
                "cannot SET `id`: a string `id` is the element's identity and is fixed at \
                 creation — insert a new element with the new id instead",
            )),
            FAULT_CMP_TEMPORAL => Err(CodeError::new(
                ErrorCode::InvalidValue,
                "cannot order-compare a temporal value with a non-temporal value; tag the \
                 literal (e.g. DATE '2024-01-01') or CAST it to the matching type",
            )),
            FAULT_DATE_PART => Err(CodeError::new(
                ErrorCode::InvalidValue,
                "_year()/_month()/_day()/_hour()/_minute()/_second() require a temporal value \
                 that carries that component (a date carries year/month/day; a time carries \
                 hour/minute/second) — a string is NOT coerced; wrap it with \
                 date()/local_datetime()/local_time() first",
            )),
            FAULT_BAD_LABEL => Err(CodeError::new(
                ErrorCode::InvalidGraphOp,
                "INSERT: a node's label expression must be a plain conjunction (`A` or `A&B`) and an edge must carry exactly one type — a disjunction/negation/wildcard or a typeless edge is not creatable",
            )),
            FAULT_CONSTRAINT => Err(CodeError::new(
                ErrorCode::ConstraintViolation,
                "write would duplicate a value under a unique constraint (use _MERGE to upsert)",
            )),
            FAULT_MERGE_KEY => Err(CodeError::new(
                ErrorCode::InvalidGraphOp,
                "_MERGE could not determine a unique key from the pattern — declare a unique constraint on the label (or narrow an ambiguous one)",
            )),
            FAULT_MERGE_EDGE => Err(CodeError::new(
                ErrorCode::NotImplemented,
                "_MERGE multi-hop compound patterns are not yet supported (v2)",
            )),
            FAULT_REQUIRED => Err(CodeError::new(
                ErrorCode::ConstraintViolation,
                "write violates a required-property constraint (a required key is missing, null, or being removed)",
            )),
            FAULT_TYPE_CONSTRAINT => Err(CodeError::new(
                ErrorCode::ConstraintViolation,
                "write violates a type constraint (a value is not of the declared scalar type)",
            )),
            FAULT_UNKNOWN_FN => {
                // Name the offending function(s) (as TS does), e.g.
                // "...: frobnicate()" — the plan collected them at lower time.
                let msg = if self.unknown_fns.is_empty() {
                    "call to an unknown or unimplemented function".to_string()
                } else {
                    let names = self
                        .unknown_fns
                        .iter()
                        .map(|n| format!("{n}()"))
                        .collect::<Vec<_>>()
                        .join(", ");

                    format!("call to an unknown or unimplemented function: {names}")
                };

                Err(CodeError::new(ErrorCode::UnknownFunction, msg))
            }
            _ => Ok(()),
        }
    }

    fn faulted(&self) -> bool {
        self.fault.load(AtomOrdering::Relaxed) != FAULT_NONE
    }
}

/// Coerce an arithmetic operand: a number passes, NULL propagates (`None`), and
/// anything else is an ISO type error recorded in `ctx` (returns `None` so eval
/// can continue to the row boundary, where the fault surfaces).
fn arith_num(v: &Val, ctx: &Ctx) -> Option<f64> {
    match v {
        Val::Null => None,
        Val::Num(n) => Some(*n),
        _ => {
            ctx.set_fault(FAULT_TYPE);
            None
        }
    }
}

fn resolve_ctx<'a>(graph: &Graph, plan: &'a CQuery, params: &'a [Val]) -> Ctx<'a> {
    Ctx {
        params,
        prop_keys: plan
            .key_names
            .iter()
            .map(|n| (graph.props.keys.get(n), graph.edge_props.keys.get(n)))
            .collect(),
        labels: plan
            .label_names
            .iter()
            .map(|n| (graph.labels.get(n), graph.etype.get(n)))
            .collect(),
        label_names: &plan.label_names,
        unknown_fns: &plan.unknown_fns,
        fault: AtomicU8::new(FAULT_NONE),
        edge_marks_pool: MarksPool::default(),
    }
}

/// Coerce a `$param` bound value (already validated by `check_count_params`) to a
/// concrete count. Defaults to 0 for anything not a finite non-negative integer —
/// unreachable after the up-front check, but keeps the accessor total/infallible.
fn count_param_val(v: Option<&Val>) -> usize {
    match v {
        Some(Val::Num(n)) if n.is_finite() && n.fract() == 0.0 && *n >= 0.0 => *n as usize,
        _ => 0,
    }
}

impl CProjection {
    /// Effective `SKIP` / `OFFSET`, resolving a dynamic `$param` bound; 0 if absent.
    fn skip_val(&self, ctx: &Ctx) -> usize {
        match &self.skip {
            None => 0,
            Some(CCount::Lit(n)) => *n,
            Some(CCount::Param(slot)) => count_param_val(ctx.params.get(*slot)),
        }
    }

    /// Effective `LIMIT`, resolving a dynamic `$param` bound; `None` if absent.
    fn limit_val(&self, ctx: &Ctx) -> Option<usize> {
        match &self.limit {
            None => None,
            Some(CCount::Lit(n)) => Some(*n),
            Some(CCount::Param(slot)) => Some(count_param_val(ctx.params.get(*slot))),
        }
    }
}

/// Resolve a bare `CCount` (literal or `$param`) to a row count. `CProjection`'s
/// `skip_val`/`limit_val` do the same for the paging carried ON a projection; this
/// is the statement-position form (`CClause::Page`).
fn count_of(c: Option<&CCount>, ctx: &Ctx) -> Option<usize> {
    match c? {
        CCount::Lit(n) => Some(*n),
        CCount::Param(slot) => Some(count_param_val(ctx.params.get(*slot))),
    }
}

/// Evaluate a compiled VALIDATOR predicate against a single graph element
/// (`Val::Node` or `Val::Edge`) bound to slot 0, with empty params. Returns the
/// ISO three-valued result — `Some(true)` / `Some(false)` / `None` (UNKNOWN/NULL)
/// — computed by the *same* evaluator a `WHERE` clause uses, so a validator and a
/// `WHERE` agree bit-for-bit with the TS engine. SQL-`CHECK` callers reject only
/// on `Some(false)`; `None` passes. An evaluation fault (e.g. an unknown function
/// or a data exception in the predicate) surfaces as `Err`, exactly as it would
/// in a query — mirroring the TS side, where the compiled closure throws.
pub fn eval_predicate(graph: &Graph, pred: &CPredicate, element: Val) -> CodeResult<Option<bool>> {
    let ctx = Ctx {
        params: &[],
        prop_keys: pred
            .key_names
            .iter()
            .map(|n| (graph.props.keys.get(n), graph.edge_props.keys.get(n)))
            .collect(),
        labels: pred
            .label_names
            .iter()
            .map(|n| (graph.labels.get(n), graph.etype.get(n)))
            .collect(),
        label_names: &pred.label_names,
        unknown_fns: &pred.unknown_fns,
        fault: AtomicU8::new(FAULT_NONE),
        edge_marks_pool: MarksPool::default(),
    };
    let mut binding = Binding::default();
    binding.set(0, element);
    let env = Env::new(graph, &ctx, &binding);
    let v = eval(&env, &pred.expr);
    ctx.check_fault()?;

    Ok(as_truth(&v))
}

/// A/B toggle for the expression VM at the hot per-row sites. Flip to `true`
/// to route those sites through the compiled stack-machine [`Program`]; `false`
/// uses the tree-walking `eval`. Both forms are kept side by side per item
/// (`CReturnItem` holds `expr` + `prog`, `CClause` holds `where_` + `where_prog`).
///
/// Measured (52k/225k graph, same-session, cooled — VM on vs off):
///   - single small expr/row (project one col, simple predicate): VM ~12-17% SLOWER
///   - many small exprs/row (4-col projection): VM ~17% SLOWER
///   - one deep predicate/row (expr-heavy filter): VM ~6% FASTER
///   - traversal/output-bound (joins, var-length): unaffected
///
/// Net: the naive scalar stack VM loses. Per-invocation setup + operand-stack
/// traffic of fat `Val`s outweighs the dispatch saved, except for a single deep
/// expression where the flat op-stream beats recursive boxed-tree pointer-chasing.
/// The win that would actually pay off is *vectorized* eval (one op over a batch
/// of rows, amortizing dispatch N-fold) — the columnar direction, not this.
const USE_VM: bool = false;

/// Toggle for the vectorized (batched, column-at-a-time) scan path. When on,
/// the single isolated-node shape `MATCH (n:L …) [WHERE pred] RETURN …` is
/// evaluated one *operation across all matched rows* instead of per row, so
/// numeric property reads gather straight from a typed `Column` and arithmetic /
/// comparison run tight `f64` loops the compiler can autovectorize. Anything
/// outside the supported numeric subset falls back to scalar `eval` per column.
///
/// Measured (52k/225k graph, same-session, cooled — vec on vs scalar off):
///   - expr-heavy numeric filter (RETURN count): 7.09ms → 1.43ms  (5.0x)
///   - ORDER BY input key, no LIMIT (50k sort):  7.24ms → 1.66ms  (4.4x)
///   - grouped aggregate (n.dept, count, avg):   3.02ms → 0.79ms  (3.8x)
///   - grouped aggregate, 2 keys:                4.08ms → 1.35ms  (3.0x)
///   - scan + numeric filter count:              1.49ms → 0.39ms  (3.8x)
///   - count(*) / count+pred:                    ~3-4x
///   - expr-heavy numeric projection (4 cols):   9.46ms → 4.42ms  (2.1x)
///   - numeric single-col projection:            1.16ms → 0.46ms  (2.5x)
///   - numeric projection over a 1-hop join:     2.78ms → 1.53ms  (1.8x)
///   - count+WHERE over a 1-hop join:            2.09ms → 1.24ms  (1.7x)
///   - DISTINCT over typed Props (raw-id group):  7.78ms → 1.23ms  (6.3x, 2 col)
///   - WITH carry + filter + project (pipeline): 4.26ms → 1.02ms  (4.2x)
///   - WITH … MATCH expand from a carried var:   7.72ms → 1.65ms  (4.7x)
///   - WITH aggregate then filter:               2.37ms → 0.92ms  (2.6x)
///   - ORDER BY input key + small LIMIT:         1.94ms → 1.37ms  (1.4x)
///   - var-length / subqueries / pure count over a join: not engaged
///   - ORDER BY on an output alias / grouped-or-DISTINCT+ORDER BY: not engaged
///
/// A read-only `MATCH … WITH … RETURN` chain runs fully vectorized end-to-end via
/// `vectorized_linear`: one columnar frame threads stage-to-stage, carrying
/// element columns forward (so prop reads / filters / ORDER BY past a `WITH` stay
/// vectorized) and adding computed value columns beside them — no per-stage
/// round-trip through `Vec<Binding>`. A `MATCH` after a `WITH` expands the frame
/// from a carried element column (`expand_frame`), fanning each row out to its
/// matching neighbors while replicating the other columns. It bails (→ scalar
/// `run_linear`) on a `WITH` that aggregates / DISTINCT / ORDER BYs mid-pipeline,
/// an expanding MATCH from an unbound/fresh start (cartesian), mutations, or
/// subqueries. `ScanCols::vals` is what makes this work: a carried node stays a
/// fast `Elem` column while `n.age * 2 AS x` rides alongside as a value column.
/// Same expressions where the scalar *bytecode VM* lost 6-17% (see [`USE_VM`])
/// win 2-5x here: dispatch amortizes over N rows, the f64 loops vectorize, and
/// values never get boxed into `Val` per row.
///
/// Tradeoffs found & handled (where vectorizing would cost more than it saves):
///   - small `LIMIT` with no WHERE: vectorizing the whole scan loses the scalar
///     streaming early-exit — `build_scan` caps the gather at `skip+limit`.
///   - an isolated-node scan: built by a tight label-bucket loop, not the general
///     matcher (which is ~3x slower per row and dominates a pure scan).
///   - a *pure* aggregate over a traversal (no WHERE): stays scalar — the scalar
///     engine stream-folds the join without materializing it, and there's no
///     per-row expression to vectorize. (With a WHERE or a projection, the
///     batched build in `build_scan` pays off.)
const USE_VEC: bool = true;

/// Whether the vectorized scan is enabled. Normally the [`USE_VEC`] const, but in
/// test builds a thread-local override lets a differential test drive the SAME
/// query through both the vectorized and scalar engines and assert they agree
/// (see `set_vec_override` / `with_vec_override`). Zero cost in release: the
/// `#[cfg(test)]` block compiles out and this inlines to the const.
#[inline]
fn use_vec() -> bool {
    #[cfg(test)]
    {
        if let Some(v) = VEC_OVERRIDE.with(|c| c.get()) {
            return v;
        }
    }
    USE_VEC
}

#[cfg(test)]
thread_local! {
    static VEC_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Force the vectorized scan on/off for the current thread while `f` runs, then
/// restore the previous setting — so a test can execute a query under both engines
/// and compare. Test-only.
#[cfg(test)]
pub(crate) fn with_vec_override<T>(on: bool, f: impl FnOnce() -> T) -> T {
    let prev = VEC_OVERRIDE.with(|c| c.replace(Some(on)));
    let out = f();
    VEC_OVERRIDE.with(|c| c.set(prev));
    out
}

/// The environment an expression evaluates against.
struct Env<'a> {
    graph: &'a Graph,
    ctx: &'a Ctx<'a>,
    binding: &'a Binding,
    /// Set while folding an aggregate over its group of bindings (the rare
    /// `eval`-time aggregate path, e.g. an aggregate in WHERE).
    group: Option<&'a [Binding]>,
    /// Folded values of a projection's extracted aggregates, resolved by
    /// [`CExpr::AggRef`] when materializing a group's output row.
    agg_values: Option<&'a [Val]>,
}

impl<'a> Env<'a> {
    fn new(graph: &'a Graph, ctx: &'a Ctx<'a>, binding: &'a Binding) -> Self {
        Env {
            graph,
            ctx,
            binding,
            group: None,
            agg_values: None,
        }
    }
}

// --- value helpers -----------------------------------------------------------

fn is_nullish(v: &Val) -> bool {
    matches!(v, Val::Null)
}

/// ISO Kleene truth: `None` = UNKNOWN. Mirrors TS `asTruth` (`Boolean(v)`).
type Truth = Option<bool>;
fn as_truth(v: &Val) -> Truth {
    match v {
        Val::Null => None,
        Val::Bool(b) => Some(*b),
        Val::Num(n) => Some(*n != 0.0 && !n.is_nan()),
        Val::Str(s) => Some(!s.is_empty()),
        _ => Some(true),
    }
}
fn not3(t: Truth) -> Truth {
    t.map(|b| !b)
}
fn and3(a: Truth, b: Truth) -> Truth {
    if a == Some(false) || b == Some(false) {
        return Some(false);
    }
    if a.is_none() || b.is_none() {
        None
    } else {
        Some(true)
    }
}
fn or3(a: Truth, b: Truth) -> Truth {
    if a == Some(true) || b == Some(true) {
        return Some(true);
    }
    if a.is_none() || b.is_none() {
        None
    } else {
        Some(false)
    }
}
fn xor3(a: Truth, b: Truth) -> Truth {
    match (a, b) {
        (Some(x), Some(y)) => Some(x != y),
        _ => None,
    }
}

/// JS `Number(v)` for the cases that matter; `None` only for nullish.
fn num_of(v: &Val) -> Option<f64> {
    match v {
        Val::Null => None,
        Val::Num(n) => Some(*n),
        Val::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Val::Str(s) => {
            let t = s.trim();
            Some(if t.is_empty() {
                0.0
            } else {
                // Reject the non-finite spellings ("inf"/"infinity"/"nan"/"Infinity")
                // that Rust's f64::from_str accepts — the TS engine's strict
                // FINITE_NUMERIC grammar coerces them to NaN, so we must too. (Hex/
                // octal already fail to parse → NaN, matching TS.)
                t.parse::<f64>()
                    .ok()
                    .filter(|n| n.is_finite())
                    .unwrap_or(f64::NAN)
            })
        }
        _ => Some(f64::NAN),
    }
}

fn js_num(n: f64) -> String {
    if n.is_nan() {
        "NaN".to_string()
    } else if n.is_infinite() {
        if n > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        }
    } else {
        // ECMA-262 `Number::toString` (via the shared jsonfmt formatter): switches to
        // exponential for |n| >= 1e21 or 0 < |n| < 1e-6 and normalizes -0 → "0" —
        // matching JS `String(n)`. Rust's Display (`format!("{n}")`) never uses
        // exponential, so `toString(1e21)`/list-map stringify diverged from the TS engine.
        crate::jsonfmt::js_number(n)
    }
}

/// JS `String(v)` for non-null values (concat/string fns guard nullish first).
fn js_str(graph: &Graph, v: &Val) -> String {
    match v {
        // Gremlin-only variants (`Val` and `GVal` are one type — see
        // `crate::value`). No GQL expression can produce one; rendering them as
        // null keeps this total rather than panicking.
        Val::Map(_) | Val::Property(_) => "null".to_string(),
        Val::Null => "null".to_string(),
        Val::Bool(b) => b.to_string(),
        Val::Num(n) => js_num(*n),
        Val::Str(s) => s.to_string(),
        Val::Temporal(t) => t.format(),
        Val::Node(i) => graph.vid.text(*i).to_string(),
        // The external edge id (like `Val::Node` above and `element_id`), which for
        // an id-less edge is the canonical `e{index}` — so this is a no-op there and
        // a fix only when the edge carries an assigned id (was always the synthetic
        // index, inconsistent with the node case).
        Val::Edge(i) => graph.edge_id(*i).into_owned(),
        Val::List(items) => items
            .iter()
            // JS `Array.prototype.join` renders a null/undefined element as the empty
            // string (`String([1,null,3])` === "1,,3"), unlike a top-level
            // `String(null)` === "null". Match it so a list→string is byte-identical.
            .map(|x| match x {
                Val::Null => String::new(),
                _ => js_str(graph, x),
            })
            .collect::<Vec<_>>()
            .join(","),
        Val::Path(p) => {
            let (vertices, edges) = (&p.vertices, &p.edges);
            // Stringify like the interleaved element sequence (vertex, edge, …).
            let mut parts = Vec::with_capacity(vertices.len() + edges.len());
            for (i, &v) in vertices.iter().enumerate() {
                if i > 0 {
                    parts.push(js_str(graph, &Val::Edge(edges[i - 1])));
                }

                parts.push(js_str(graph, &Val::Node(v)));
            }

            parts.join(",")
        }
        // A record stringifies to its canonical JSON object (via the shared result
        // serializer, so it's byte-identical to how the map serializes elsewhere).
        Val::Record(_) => {
            let mut s = String::new();
            crate::codec::push_value(&mut s, &val_to_value(graph, v));
            s
        }
    }
}

/// Make a `Val::Str` from anything that can produce an owned/borrowed `str`.
fn vstr(s: impl Into<Arc<str>>) -> Val {
    Val::Str(s.into())
}

/// Look up field `key` in a record/map — a binary search (keys are canonical /
/// sorted). A missing field reads as NULL (an absent field, three-valued like an
/// absent property). Cheap clone: scalars copy, `Str`/`Map` are refcount bumps.
fn map_get(pairs: &[(Arc<str>, Val)], key: &str) -> Val {
    match pairs.binary_search_by(|(k, _)| k.as_ref().cmp(key)) {
        Ok(i) => pairs[i].1.clone(),
        Err(_) => Val::Null,
    }
}

/// Structural / identity equality (Null == Null is true). Used by `=` (after a
/// nullish guard) and by the element-pattern predicate's strict comparison.
fn val_eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Null, Val::Null) => true,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::Num(x), Val::Num(y)) => x == y,
        (Val::Str(x), Val::Str(y)) => x == y,
        // Distinct kinds (date vs datetime) are never equal (enum inequality).
        (Val::Temporal(x), Val::Temporal(y)) => x == y,
        (Val::Node(x), Val::Node(y)) => x == y,
        (Val::Edge(x), Val::Edge(y)) => x == y,
        (Val::List(x), Val::List(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| val_eq(p, q))
        }
        // Records are equal iff they have the same fields (keys are canonical, so
        // positional) with recursively-equal values. ISO records support `=`/`<>`.
        (Val::Record(x), Val::Record(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|((k1, v1), (k2, v2))| k1 == k2 && val_eq(v1, v2))
        }
        _ => false,
    }
}

/// Push `v` into `out` unless an equal element is already present (structural
/// equality, first occurrence wins). The building block for the ISO GQL set-style
/// list functions (`list_union`/`intersection`/`difference`), all of which dedup.
fn push_unique(out: &mut Vec<Val>, v: &Val) {
    if !out.iter().any(|x| val_eq(x, v)) {
        out.push(v.clone());
    }
}

/// Whether `a` and `b` are the same orderable primitive type (number, string,
/// or boolean). ISO ordering (`< > <= >=`) is only defined within such a type;
/// across types — or for graph elements — the comparison is UNKNOWN, not a
/// coerced bool. (Mirrors the TS executor's `orderable` guard.)
fn orderable_pair(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Num(_), Val::Num(_)) | (Val::Str(_), Val::Str(_)) | (Val::Bool(_), Val::Bool(_)) => {
            true
        }
        // Instants (date/datetime, same kind) are relationally orderable;
        // durations and cross-kind pairs are not (`rel_cmp` → None).
        (Val::Temporal(x), Val::Temporal(y)) => x.rel_cmp(y).is_some(),
        _ => false,
    }
}

/// Partial ordering for the relational operators `< > <= >=`. `None` =
/// incomparable (different types, or a graph element) → the operator yields
/// UNKNOWN, never a coerced bool. This is NOT the sort order — see [`cmp_total`].
fn val_cmp(a: &Val, b: &Val) -> Option<Ordering> {
    match (a, b) {
        (Val::Num(x), Val::Num(y)) => x.partial_cmp(y),
        (Val::Str(x), Val::Str(y)) => Some(x.cmp(y)),
        (Val::Bool(x), Val::Bool(y)) => Some(x.cmp(y)),
        (Val::Temporal(x), Val::Temporal(y)) => x.rel_cmp(y),
        (Val::Node(x), Val::Node(y)) => Some(x.cmp(y)),
        (Val::Edge(x), Val::Edge(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// Type-group rank for the TOTAL sort order (mirrors the TS `typeRank`):
/// number < string < boolean < other (graph elements / lists). Null is handled
/// by [`cmp_total`] before this is consulted.
fn type_rank(v: &Val) -> u8 {
    match v {
        Val::Num(_) => 0,
        Val::Str(_) => 1,
        Val::Bool(_) => 2,
        Val::Temporal(_) => 3,
        _ => 4,
    }
}

/// A TOTAL order across value types, used by ORDER BY / min / max / list_sort so
/// a mixed-type column sorts deterministically (unlike `val_cmp`, which is
/// partial). Byte-for-byte identical to the TS `compareValues`: null sorts
/// largest; otherwise different type groups order by `type_rank`, and within a
/// group numbers/strings/booleans compare naturally while two graph
/// elements/lists compare Equal (leaving them in stable order). NaN, like the
/// relational path, compares Equal to every number.
fn cmp_total(a: &Val, b: &Val) -> Ordering {
    let a_null = is_nullish(a);
    let b_null = is_nullish(b);
    if a_null && b_null {
        return Ordering::Equal;
    }
    if a_null {
        return Ordering::Greater;
    }
    if b_null {
        return Ordering::Less;
    }
    let (ra, rb) = (type_rank(a), type_rank(b));
    if ra != rb {
        return ra.cmp(&rb);
    }
    match (a, b) {
        (Val::Num(x), Val::Num(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (Val::Str(x), Val::Str(y)) => x.cmp(y),
        (Val::Bool(x), Val::Bool(y)) => x.cmp(y),
        (Val::Temporal(x), Val::Temporal(y)) => x.cmp_total(y),
        // Lists compare element-wise (lexicographic, shorter-is-less on a prefix),
        // recursing through the same total order — so `min`/`max` and `ORDER BY`
        // over list values are well-defined and match the TS `compareValues`.
        (Val::List(x), Val::List(y)) => {
            for (xi, yi) in x.iter().zip(y.iter()) {
                let c = cmp_total(xi, yi);
                if c != Ordering::Equal {
                    return c;
                }
            }
            x.len().cmp(&y.len())
        }
        // Records: keys are canonical (sorted), so compare field-by-field (key
        // then value), shorter-is-less. Gives ORDER BY / DISTINCT a total order
        // even though ISO defines no relational `<`/`>` on records.
        (Val::Record(x), Val::Record(y)) => {
            for ((k1, v1), (k2, v2)) in x.iter().zip(y.iter()) {
                let kc = k1.cmp(k2);
                if kc != Ordering::Equal {
                    return kc;
                }
                let vc = cmp_total(v1, v2);
                if vc != Ordering::Equal {
                    return vc;
                }
            }
            x.len().cmp(&y.len())
        }
        _ => Ordering::Equal,
    }
}

/// Compare two non-null operands to a three-valued result. Equality holds across
/// any types (mismatched types are simply unequal); ordering across incomparable
/// types is UNKNOWN (`Val::Null`), not a coerced bool — EXCEPT a temporal vs a
/// non-temporal relational comparison, which is a type error (an untagged string
/// param vs a stored DATE is a mistake, not "no rows"): it faults via `ctx`.
fn compare_vals(ctx: &Ctx, op: CompareOp, lv: &Val, rv: &Val) -> Val {
    match op {
        CompareOp::Eq => Val::Bool(val_eq(lv, rv)),
        CompareOp::Ne => Val::Bool(!val_eq(lv, rv)),
        _ if !orderable_pair(lv, rv) => {
            // Exactly one operand temporal → ordering it against a string/number is
            // a type error; both-non-temporal (num vs str) or both-temporal
            // cross-kind (date vs time) stay UNKNOWN as before.
            if matches!(lv, Val::Temporal(_)) != matches!(rv, Val::Temporal(_)) {
                ctx.set_fault(FAULT_CMP_TEMPORAL);
            }
            Val::Null
        }
        _ => {
            let c = val_cmp(lv, rv);
            Val::Bool(match op {
                CompareOp::Lt => c == Some(Ordering::Less),
                CompareOp::Gt => c == Some(Ordering::Greater),
                CompareOp::Le => matches!(c, Some(Ordering::Less | Ordering::Equal)),
                CompareOp::Ge => matches!(c, Some(Ordering::Greater | Ordering::Equal)),
                CompareOp::Eq | CompareOp::Ne => unreachable!(),
            })
        }
    }
}

/// `v IN list` as a three-valued OR of equalities (identity = empty → FALSE).
fn in_list(v: &Val, list: &Val) -> Truth {
    let Val::List(items) = list else { return None };
    let mut saw_unknown = false;
    for e in items.iter() {
        if is_nullish(v) || is_nullish(e) {
            saw_unknown = true;
            continue;
        }
        if val_eq(e, v) {
            return Some(true);
        }
    }
    if saw_unknown {
        None
    } else {
        Some(false)
    }
}

/// The bit pattern a number groups / dedups by, canonicalized so that grouping
/// agrees with equality.
///
/// Two values need collapsing. **NaN**: the engine's total order treats NaN ==
/// NaN, but the RAW bits differ by sign and payload depending on which operation
/// produced it — `ln(-1)` and `x / NaN` do not agree — which split one logical
/// value into several groups. **Signed zero**: `-0.0 == 0.0` is true, and this
/// engine normalizes the distinction absolutely everywhere else — `=`, `IN`,
/// ORDER BY, `sign()`, the result JSON, `to_string`, and the property index all
/// treat them as one value, and `1 / ±0` faults rather than yielding ±∞. Keeping
/// them apart HERE alone produced two DISTINCT groups whose rendered values were
/// both `0` — indistinguishable in the output, so readable only as a bug. The
/// Gremlin engine's `dedup_key` already collapsed them; this brings GQL in line.
fn group_num_bits(n: f64) -> u64 {
    if n.is_nan() {
        return f64::NAN.to_bits();
    }
    // Signed zero, BRANCHLESSLY: IEEE 754 gives `-0.0 + 0.0 == +0.0` under the
    // default rounding mode, while `x + 0.0 == x` exactly for every other finite
    // or infinite x. So one add does the normalization with no compare and no
    // branch — the historical reason for keeping -0 apart here was the cost of
    // that check, and this removes the cost rather than the correctness. (Rust
    // does not assume fast-math, so the add is never optimized away.)
    (n + 0.0).to_bits()
}

/// A canonical, hashable key for a value — grouping, DISTINCT, row keys.
fn val_key(v: &Val, out: &mut String) {
    match v {
        // Gremlin-only (see `crate::value`) — unreachable from GQL, and keyed
        // like null so the function stays total.
        Val::Map(_) | Val::Property(_) => out.push('N'),
        Val::Null => out.push('N'),
        Val::Bool(b) => {
            out.push('b');
            out.push(if *b { '1' } else { '0' });
        }
        Val::Num(n) => {
            let _ = write!(out, "n{:016x}", group_num_bits(*n));
        }
        Val::Str(s) => {
            // Raw byte push (same bytes as `write!("s{s}")`, without the fmt
            // machinery) — the scalar grouping/DISTINCT key builder is per-row hot.
            out.push('s');
            out.push_str(s);
        }
        Val::Temporal(t) => {
            let _ = write!(out, "t{}{}", t.tag(), t.format());
        }
        Val::Node(i) => {
            let _ = write!(out, "@v{i}");
        }
        Val::Edge(i) => {
            let _ = write!(out, "@e{i}");
        }
        Val::List(items) => {
            out.push('[');
            for it in items.iter() {
                val_key(it, out);
                out.push(',');
            }
            out.push(']');
        }
        // Canonical (sorted) keys → a canonical grouping/DISTINCT key string.
        Val::Record(pairs) => {
            out.push('{');
            for (k, it) in pairs.iter() {
                out.push_str(k);
                out.push(':');
                val_key(it, out);
                out.push(',');
            }
            out.push('}');
        }
        Val::Path(p) => {
            let (vertices, edges) = (&p.vertices, &p.edges);
            // Structural: two paths are the same key iff they visit the same
            // vertices in the same order via the same edges (so `DISTINCT p` works).
            out.push('P');
            for &v in vertices {
                let _ = write!(out, "v{v}");
            }
            out.push('|');
            for &e in edges {
                let _ = write!(out, "e{e}");
            }
        }
    }
}

/// Are two grouping-key value tuples the same group, WITHOUT building the string
/// key? A correct *refinement* of [`val_key`] equality: `true` ⇒ the `val_key`
/// strings are equal (safe to accumulate into the same group). It may return
/// `false` for equal-but-uncommon kinds (temporal/list/path) — that only forgoes
/// the fast path (the row falls through to the string-keyed map, still correct).
/// Used for the streaming "same as previous row" fast path in grouped aggregation,
/// where `WITH <driving-var>, <agg>` yields rows already contiguous by the key —
/// so most rows never build or hash a key string.
fn group_vals_eq(a: &[Val], b: &[Val]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| group_val_eq(x, y))
}

fn group_val_eq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Null, Val::Null) => true,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::Num(x), Val::Num(y)) => group_num_bits(*x) == group_num_bits(*y),
        (Val::Str(x), Val::Str(y)) => x == y,
        (Val::Node(x), Val::Node(y)) => x == y,
        (Val::Edge(x), Val::Edge(y)) => x == y,
        // Temporal/List/Path: defer to the slow (string-keyed) path — comparing
        // them here can allocate (temporal format) or recurse, and they are rare
        // grouping keys. Returning `false` is always correctness-safe.
        _ => false,
    }
}

fn row_key(b: &Binding) -> String {
    let mut s = String::new();
    for cell in &b.0 {
        match cell {
            Some(v) => val_key(v, &mut s),
            None => s.push('\u{2}'), // distinct marker for an unbound slot
        }
        s.push('\u{1}');
    }
    s
}

/// Canonical key for an output [`Value`] cell (DISTINCT / set-op identity).
fn value_key(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push('N'),
        Value::Bool(b) => {
            out.push('b');
            out.push(if *b { '1' } else { '0' });
        }
        Value::Num(n) => {
            let _ = write!(out, "n{:016x}", group_num_bits(*n));
        }
        Value::Str(s) => {
            let _ = write!(out, "s{s}");
        }
        Value::Temporal(t) => {
            let _ = write!(out, "t{}{}", t.tag(), t.format());
        }
        Value::List(items) => {
            out.push('[');
            for it in items {
                value_key(it, out);
                out.push(',');
            }
            out.push(']');
        }
        Value::Map(pairs) => {
            out.push('{');
            for (k, val) in pairs {
                let _ = write!(out, "{k}=");
                value_key(val, out);
                out.push(',');
            }
            out.push('}');
        }
    }
}

fn value_row_key(row: &[Value]) -> String {
    let mut s = String::new();
    for cell in row {
        value_key(cell, &mut s);
        s.push('\u{1}');
    }
    s
}

// --- property / label access -------------------------------------------------

fn value_to_val(v: &Value) -> Val {
    Val::from_stored(v, true)
}

/// Set one field of a graph-algorithm config from a CALL config-map entry, keyed
/// by the algorithm's JSON config field name. An unknown key (with a "did you
/// mean" hint) or a value of the wrong type is an error — a silently-dropped key
/// once hid the `pivots` bug (approximate betweenness that never sampled), so the
/// config map is validated rather than best-effort. Mirrors the TS engine so both
/// fault byte-identically.
fn apply_algo_config(
    cfg: &mut crate::algo::AlgoConfig,
    field: &str,
    v: &Val,
) -> Result<(), CodeError> {
    let err = |m: String| CodeError::new(ErrorCode::InvalidValue, m);
    let want_str = |v: &Val| match v {
        Val::Str(s) => Ok(s.to_string()),
        _ => Err(err(format!("config key '{field}' expects a string"))),
    };
    // A strict number — NOT `num_of` (which coerces strings/bools engine-wide);
    // a config value should be a genuine number, matching the TS `typeof` check.
    let want_num = |v: &Val| match v {
        Val::Num(n) => Ok(*n),
        _ => Err(err(format!("config key '{field}' expects a number"))),
    };
    // A list config value (personalized-PageRank seed set): keep its string
    // elements, exactly as the JSON-config path does.
    let want_strs = |v: &Val| match v {
        Val::List(items) => Ok(items
            .iter()
            .filter_map(|x| match x {
                Val::Str(s) => Some(s.to_string()),
                _ => None,
            })
            .collect()),
        _ => Err(err(format!("config key '{field}' expects a list"))),
    };
    let want_bool = |v: &Val| match v {
        Val::Bool(b) => Ok(*b),
        _ => Err(err(format!("config key '{field}' expects a boolean"))),
    };
    match field {
        "edgeLabel" => cfg.edge_label = Some(want_str(v)?),
        "direction" => cfg.direction = Some(want_str(v)?),
        "weightProperty" => cfg.weight_property = Some(want_str(v)?),
        "dampingFactor" => cfg.damping_factor = Some(want_num(v)?),
        "iterations" => cfg.iterations = Some(want_num(v)? as u32),
        "pivots" => cfg.pivots = Some(want_num(v)? as u32),
        "seedProperty" => cfg.seed_property = Some(want_str(v)?),
        "source" => cfg.source = Some(want_str(v)?),
        "sourceNodes" => cfg.source_nodes = Some(want_strs(v)?),
        "target" => cfg.target = Some(want_str(v)?),
        "writeProperty" => cfg.write_property = Some(want_str(v)?),
        "algorithm" => cfg.algorithm = Some(want_str(v)?),
        "heuristicProperty" => cfg.heuristic_property = Some(want_str(v)?),
        "feature" => cfg.feature = Some(want_str(v)?),
        "op" => cfg.op = Some(want_str(v)?),
        "includeSelf" => cfg.include_self = Some(want_bool(v)?),
        "norm" => cfg.norm = Some(want_str(v)?),
        _ => {
            return Err(err(match crate::algo::suggest_config_key(field) {
                Some(s) => format!("unknown config key '{field}' (did you mean '{s}'?)"),
                None => format!("unknown config key '{field}'"),
            }))
        }
    }
    Ok(())
}

/// A store element's present properties as a sorted `Value::Map` — the shape a
/// returned node/edge's `properties` field serializes to. Keys are sorted so the
/// object is deterministic (the columnar store has no per-element key order).
fn props_map(store: &crate::graph::Properties, strs: &crate::graph::Dict, idx: usize) -> Value {
    let mut props: Vec<(Arc<str>, Value)> = (0..store.keys.len() as u32)
        .filter(|&kid| store.is_present_id(idx, kid))
        // `keys.arc` shares the interned key Arc (refcount bump) instead of
        // `Arc::from(text)` allocating a fresh copy per key on every element.
        .map(|kid| (store.keys.arc(kid), store.value_id(idx, kid, strs)))
        .collect();
    props.sort_by(|a, b| a.0.cmp(&b.0));
    Value::Map(props)
}

/// Process-lifetime interned `Arc<str>`s for the constant node/edge map keys, so
/// serializing an element clones them (a refcount bump) instead of re-allocating
/// `"id"`/`"labels"`/… on every element `val_to_value` emits.
struct ElemKeys {
    id: Arc<str>,
    labels: Arc<str>,
    properties: Arc<str>,
    from: Arc<str>,
    to: Arc<str>,
    vertices: Arc<str>,
    edges: Arc<str>,
    length: Arc<str>,
}
fn elem_keys() -> &'static ElemKeys {
    static K: std::sync::OnceLock<ElemKeys> = std::sync::OnceLock::new();
    K.get_or_init(|| ElemKeys {
        id: Arc::from("id"),
        labels: Arc::from("labels"),
        properties: Arc::from("properties"),
        from: Arc::from("from"),
        to: Arc::from("to"),
        vertices: Arc::from("vertices"),
        edges: Arc::from("edges"),
        length: Arc::from("length"),
    })
}

/// Project a runtime value to the core output [`Value`]. A returned node/edge
/// reference serializes to a `{id, labels, properties}` object (matching the TS
/// engine) so `RETURN n` is useful, not a bare id.
/// The canonical result `Value::Map` for a vertex (`{id, labels, properties}`) —
/// exposed so the Gremlin engine serializes an element byte-identically to GQL.
pub(crate) fn node_result_value(graph: &Graph, i: u32) -> Value {
    val_to_value(graph, &Val::Node(i))
}

/// The canonical result `Value::Map` for an edge (`{id, from, to, labels,
/// properties}`) — see [`node_result_value`].
pub(crate) fn edge_result_value(graph: &Graph, i: u32) -> Value {
    val_to_value(graph, &Val::Edge(i))
}

fn val_to_value(graph: &Graph, v: &Val) -> Value {
    match v {
        // Gremlin-only (see `crate::value`) — unreachable from GQL.
        Val::Map(_) | Val::Property(_) => Value::Null,
        Val::Null => Value::Null,
        Val::Bool(b) => Value::Bool(*b),
        Val::Num(n) => Value::Num(*n),
        Val::Str(s) => Value::Str(s.clone()), // shared Arc — refcount bump, no alloc
        Val::Temporal(t) => Value::Temporal(*t),
        Val::List(items) => Value::List(items.iter().map(|x| val_to_value(graph, x)).collect()),
        // A runtime record → the result map (keys already canonical/sorted).
        Val::Record(pairs) => Value::Map(
            pairs
                .iter()
                .map(|(k, v)| (k.clone(), val_to_value(graph, v)))
                .collect(),
        ),
        Val::Node(i) => {
            let mut labels: Vec<Arc<str>> = graph
                .vertex_labels(*i)
                .iter()
                .map(|&l| graph.labels.arc(l))
                .collect();
            labels.sort_unstable();
            let k = elem_keys();
            Value::Map(vec![
                (k.id.clone(), Value::Str(graph.vid.arc(*i))),
                (
                    k.labels.clone(),
                    Value::List(labels.into_iter().map(Value::Str).collect()),
                ),
                (
                    k.properties.clone(),
                    props_map(&graph.props, &graph.strs, *i as usize),
                ),
            ])
        }
        Val::Edge(i) => {
            let idx = *i as usize;
            let k = elem_keys();
            Value::Map(vec![
                (
                    k.id.clone(),
                    Value::Str(Arc::from(graph.edge_id(*i).as_ref())),
                ),
                (k.from.clone(), Value::Str(graph.vid.arc(graph.e_src[idx]))),
                (k.to.clone(), Value::Str(graph.vid.arc(graph.e_dst[idx]))),
                (
                    k.labels.clone(),
                    Value::List(vec![Value::Str(graph.etype.arc(graph.e_type[idx]))]),
                ),
                (
                    k.properties.clone(),
                    props_map(&graph.edge_props, &graph.strs, idx),
                ),
            ])
        }
        Val::Path(p) => {
            let (vertices, edges) = (&p.vertices, &p.edges);
            // `{vertices, edges, length}` — the vertices/edges reuse the element
            // serialization above; `length` is the hop (edge) count. Mirrors the
            // TS `Path.toJSON()` byte-for-byte (field order, sorted labels/props).
            let k = elem_keys();
            Value::Map(vec![
                (
                    k.vertices.clone(),
                    Value::List(
                        vertices
                            .iter()
                            .map(|&v| val_to_value(graph, &Val::Node(v)))
                            .collect(),
                    ),
                ),
                (
                    k.edges.clone(),
                    Value::List(
                        edges
                            .iter()
                            .map(|&e| val_to_value(graph, &Val::Edge(e)))
                            .collect(),
                    ),
                ),
                (k.length.clone(), Value::Num(edges.len() as f64)),
            ])
        }
    }
}

/// ISO: an absent property — or a property of a non-element/NULL — yields NULL.
/// Vertices and edges read from the same columnar store; `key_ref`'s id was
/// resolved once at execute time (no per-access name lookup).
/// Trim `s` from the chosen ends. `chars = Some(set)` strips any code point in
/// that set; `None` strips Unicode whitespace (matching `str::trim*`). Code-point
/// based (`chars()` / TS `[...s]`) so it's byte-identical across engines.
fn multi_trim(s: &str, chars: Option<&str>, leading: bool, trailing: bool) -> String {
    let cps: Vec<char> = s.chars().collect();
    let hit = |ch: char| match chars {
        Some(set) => set.chars().any(|c| c == ch),
        None => ch.is_whitespace(),
    };
    let mut lo = 0;
    let mut hi = cps.len();
    if leading {
        while lo < hi && hit(cps[lo]) {
            lo += 1;
        }
    }
    if trailing {
        while hi > lo && hit(cps[hi - 1]) {
            hi -= 1;
        }
    }
    cps[lo..hi].iter().collect()
}

/// The `trim`/`ltrim`/`rtrim`/`btrim` scalar arm: null in → null out; an optional
/// 2nd arg is the character set to strip (else whitespace).
fn trim_arm(a: Option<&Val>, b: Option<&Val>, graph: &Graph, leading: bool, trailing: bool) -> Val {
    match a {
        Some(v) if !is_nullish(v) => {
            let s = js_str(graph, v);
            let chars = b.filter(|c| !is_nullish(c)).map(|c| js_str(graph, c));
            vstr(multi_trim(&s, chars.as_deref(), leading, trailing))
        }
        _ => Val::Null,
    }
}

/// The graph-element predicates. All are three-valued: a null operand — or a
/// type mismatch (a non-edge for `DIRECTED`, a non-node/edge for `SOURCE OF`) —
/// yields NULL rather than a definite bool. `SOURCE/DESTINATION OF` reads the
/// edge's stored endpoints (`e_src`/`e_dst`); `ALL_DIFFERENT`/`SAME` compare by
/// value/element identity (`val_eq`, which is element-id for nodes/edges).
fn eval_graph_pred(
    env: &Env,
    kind: super::ast::GraphPredKind,
    args: &[CExpr],
    negated: bool,
) -> Val {
    use super::ast::GraphPredKind::*;
    let vals: Vec<Val> = args.iter().map(|a| eval(env, a)).collect();
    let result: Option<bool> = match kind {
        Directed => match vals.first() {
            Some(Val::Edge(_)) => Some(true), // every lenke edge is directed
            _ => None,                        // null / non-edge → unknown
        },
        SourceOf | DestOf => match (vals.first(), vals.get(1)) {
            (Some(Val::Node(vi)), Some(Val::Edge(ei))) => {
                let idx = *ei as usize;
                let endpoint = if matches!(kind, SourceOf) {
                    env.graph.e_src[idx]
                } else {
                    env.graph.e_dst[idx]
                };
                Some(endpoint == *vi)
            }
            _ => None, // null / wrong kinds → unknown
        },
        AllDifferent | Same => {
            if vals.iter().any(is_nullish) {
                None
            } else if matches!(kind, Same) {
                Some(vals.windows(2).all(|w| val_eq(&w[0], &w[1])))
            } else {
                let mut distinct = true;
                'outer: for i in 0..vals.len() {
                    for j in (i + 1)..vals.len() {
                        if val_eq(&vals[i], &vals[j]) {
                            distinct = false;
                            break 'outer;
                        }
                    }
                }
                Some(distinct)
            }
        }
    };
    match result {
        Some(b) => Val::Bool(if negated { !b } else { b }),
        None => Val::Null,
    }
}

/// Does a NON-null value match a scalar type category? Numeric split: `integer` =
/// a whole-valued number, `float` = any number (lenke has one f64 numeric type —
/// boundary inference, matching how it renders whole f64s as ints / tags them
/// `g:Int64`). The open-record category isn't here — `ANY RECORD` is a
/// [`TypeTest::AnyRecord`], handled in [`value_is_typed_ty`].
fn category_matches(v: &Val, category: &str) -> bool {
    use crate::temporal::Temporal;
    match category {
        "any" => true,
        "null" => false, // v is non-null
        "bool" => matches!(v, Val::Bool(_)),
        "string" => matches!(v, Val::Str(_)),
        "integer" => matches!(v, Val::Num(n) if n.is_finite() && n.fract() == 0.0),
        "float" => matches!(v, Val::Num(_)),
        "list" => matches!(v, Val::List(_)),
        "date" => matches!(v, Val::Temporal(Temporal::Date(_))),
        "local_time" => matches!(v, Val::Temporal(Temporal::Time(_))),
        "local_datetime" => matches!(v, Val::Temporal(Temporal::DateTime(_))),
        "zoned_time" => matches!(v, Val::Temporal(Temporal::ZonedTime(_))),
        "zoned_datetime" => matches!(v, Val::Temporal(Temporal::ZonedDateTime(_))),
        "duration" => matches!(v, Val::Temporal(Temporal::Duration(_))),
        _ => false,
    }
}

/// The ISO value-type predicate `x IS TYPED <value type> [NOT NULL]`. Null conforms
/// to any *nullable* type (Neo4j-verified reading), so a null value is `!not_null`
/// regardless of type. A closed `RECORD {…}` is CLOSED on extras and matches each
/// present field's value against its type (a field null is OK unless the field is
/// `NOT NULL`; an absent field is OK unless `NOT NULL`) — mirrors the record
/// constraint's `value_matches`, but keeps the predicate's scalar vocabulary.
fn value_is_typed_ty(v: &Val, ty: &TypeTest, not_null: bool) -> bool {
    if is_nullish(v) {
        return !not_null;
    }
    match ty {
        TypeTest::Scalar(category) => category_matches(v, category),
        TypeTest::AnyRecord => matches!(v, Val::Record(_)),
        TypeTest::Record(fields) => {
            let Val::Record(pairs) = v else { return false };
            // Closed: every present key must be a declared field.
            if pairs.iter().any(|(vk, _)| {
                fields
                    .binary_search_by(|(fk, _, _)| fk.as_str().cmp(vk.as_ref()))
                    .is_err()
            }) {
                return false;
            }
            fields.iter().all(|(fk, ft, field_not_null)| {
                match pairs.binary_search_by(|(vk, _)| vk.as_ref().cmp(fk.as_str())) {
                    Ok(i) => value_is_typed_ty(&pairs[i].1, ft, *field_not_null),
                    Err(_) => !field_not_null, // absent OK unless the field is NOT NULL
                }
            })
        }
    }
}

/// `PROPERTY_EXISTS(n, key)`: is `key` a *present* property of element `n`? A
/// `Bool` for an element (distinguishing an absent key from a stored null), and
/// `Null` for a non-element/NULL (three-valued, like a comparison). Resolves the
/// key exactly like [`prop_of`] but tests presence instead of reading the value.
fn prop_present(graph: &Graph, ctx: &Ctx, bound: &Val, key_ref: usize) -> Val {
    let (store, kid, idx) = match bound {
        Val::Node(vi) => (&graph.props, ctx.prop_keys[key_ref].0, *vi as usize),
        Val::Edge(ei) => (&graph.edge_props, ctx.prop_keys[key_ref].1, *ei as usize),
        _ => return Val::Null,
    };
    // `kid == None` means the key was never interned in this store → not present.
    Val::Bool(kid.is_some_and(|kid| store.is_present_id(idx, kid)))
}

fn prop_of(graph: &Graph, ctx: &Ctx, bound: &Val, key_ref: usize) -> Val {
    let (store, kid, idx) = match bound {
        Val::Node(vi) => (&graph.props, ctx.prop_keys[key_ref].0, *vi as usize),
        Val::Edge(ei) => (&graph.edge_props, ctx.prop_keys[key_ref].1, *ei as usize),
        _ => return Val::Null,
    };
    let Some(kid) = kid else { return Val::Null };
    // Read the column directly: a string property is a refcount bump (Rc clone),
    // not an allocation; numbers/bools are copied; Mixed converts.
    match store.cols.get(kid as usize) {
        Some(Column::Num { data, present }) if present.get(idx) => Val::Num(data[idx]),
        Some(Column::Bool { data, present }) if present.get(idx) => Val::Bool(data[idx]),
        Some(Column::Str { data, present }) if present.get(idx) => {
            Val::Str(graph.strs.arc(data[idx]))
        }
        Some(Column::Temporal { data, present }) if present.get(idx) => {
            Val::Temporal(data.get(idx))
        }
        // A typed vector column reconstructs the same list `value_to_val` would
        // yield for the boxed form — via the zero-copy slice accessor.
        Some(Column::Vec { .. }) => store
            .vector_id(idx, kid)
            .map(|s| Val::List(s.iter().map(|x| Val::Num(*x)).collect()))
            .unwrap_or(Val::Null),
        Some(Column::Mixed { data }) => data[idx].as_ref().map(value_to_val).unwrap_or(Val::Null),
        // A de-boxed record synthesizes its map (or reads an escapee) via the store.
        Some(Column::Record { .. }) => value_to_val(&store.value_id(idx, kid, &graph.strs)),
        _ => Val::Null,
    }
}

/// Read `element.root.field.field…` — a field access on a stored record — WITHOUT
/// materializing the whole map. Borrows the stored root `Value` in place (only a
/// `Mixed`-boxed map can be navigated), walks the descent in the `Value` domain,
/// and converts ONLY the leaf to a `Val`. A non-element base, a non-map root, or
/// a missing/scalar segment → `Val::Null` (three-valued, like a missing field).
fn prop_field_of(env: &Env, var_slot: usize, root_key_ref: usize, descent: &[Arc<str>]) -> Val {
    let (store, kid, idx) = match env.binding.get(var_slot) {
        Some(Val::Node(vi)) => (
            &env.graph.props,
            env.ctx.prop_keys[root_key_ref].0,
            *vi as usize,
        ),
        Some(Val::Edge(ei)) => (
            &env.graph.edge_props,
            env.ctx.prop_keys[root_key_ref].1,
            *ei as usize,
        ),
        _ => return Val::Null,
    };
    let Some(kid) = kid else { return Val::Null };
    let segs: Vec<&str> = descent.iter().map(|s| s.as_ref()).collect();
    // `field_at` reads a de-boxed record field DIRECTLY from its sub-column (no
    // whole-map materialization) and walks a boxed `Mixed` map otherwise.
    value_to_val(&store.field_at(idx, kid, &segs, &env.graph.strs))
}

fn eval_label_node(graph: &Graph, ctx: &Ctx, vi: u32, expr: &CLabelExpr) -> bool {
    match expr {
        CLabelExpr::Label(r) => ctx.labels[*r].0.is_some_and(|lid| graph.has_label(vi, lid)),
        CLabelExpr::Wildcard => !graph.vertex_labels(vi).is_empty(),
        CLabelExpr::Not(e) => !eval_label_node(graph, ctx, vi, e),
        CLabelExpr::And(l, r) => {
            eval_label_node(graph, ctx, vi, l) && eval_label_node(graph, ctx, vi, r)
        }
        CLabelExpr::Or(l, r) => {
            eval_label_node(graph, ctx, vi, l) || eval_label_node(graph, ctx, vi, r)
        }
    }
}

fn eval_label_edge(ctx: &Ctx, etype: u32, expr: &CLabelExpr) -> bool {
    match expr {
        CLabelExpr::Label(r) => ctx.labels[*r].1 == Some(etype),
        CLabelExpr::Wildcard => true, // an edge always has exactly one type
        CLabelExpr::Not(e) => !eval_label_edge(ctx, etype, e),
        CLabelExpr::And(l, r) => eval_label_edge(ctx, etype, l) && eval_label_edge(ctx, etype, r),
        CLabelExpr::Or(l, r) => eval_label_edge(ctx, etype, l) || eval_label_edge(ctx, etype, r),
    }
}

/// `IS LABELED` over a runtime element value.
fn labels_match(graph: &Graph, ctx: &Ctx, el: &Val, expr: &CLabelExpr) -> bool {
    match el {
        Val::Node(vi) => eval_label_node(graph, ctx, *vi, expr),
        Val::Edge(ei) => eval_label_edge(ctx, graph.e_type[*ei as usize], expr),
        _ => false,
    }
}

fn matches_label(graph: &Graph, ctx: &Ctx, vi: u32, label: Option<&CLabelExpr>) -> bool {
    label.is_none_or(|e| eval_label_node(graph, ctx, vi, e))
}

// --- expression evaluation ---------------------------------------------------

fn truth_to_val(t: Truth) -> Val {
    match t {
        Some(b) => Val::Bool(b),
        None => Val::Null,
    }
}

/// One step of a left-associative arithmetic fold `lv <op> rv`. Preserves the
/// pre-refactor binary semantics: temporal arithmetic, division/modulo-by-zero
/// faulting to null, and null propagation from a non-numeric operand.
fn arith_step(env: &Env, op: ArithOp, lv: Val, rv: Val) -> Val {
    if matches!(lv, Val::Temporal(_)) || matches!(rv, Val::Temporal(_)) {
        return temporal_arith(env.ctx, op, &lv, &rv);
    }
    match (arith_num(&lv, env.ctx), arith_num(&rv, env.ctx)) {
        (Some(a), Some(b)) => {
            if matches!(op, ArithOp::Div | ArithOp::Mod) && b == 0.0 {
                env.ctx.set_fault(FAULT_DIV_ZERO);
                Val::Null
            } else {
                Val::Num(match op {
                    ArithOp::Add => a + b,
                    ArithOp::Sub => a - b,
                    ArithOp::Mul => a * b,
                    ArithOp::Div => a / b,
                    ArithOp::Mod => a % b,
                })
            }
        }
        _ => Val::Null,
    }
}

/// One step of a left-associative string/list concat fold `lv || rv`. A null
/// operand yields null; two lists concatenate; otherwise the operands stringify.
fn concat_step(env: &Env, lv: Val, rv: Val) -> Val {
    if is_nullish(&lv) || is_nullish(&rv) {
        return Val::Null;
    }
    match (&lv, &rv) {
        (Val::List(a), Val::List(b)) => Val::List(a.iter().chain(b.iter()).cloned().collect()),
        _ => vstr(js_str(env.graph, &lv) + &js_str(env.graph, &rv)),
    }
}

/// One column-at-a-time step of a left-associative arithmetic fold in the
/// vectorized evaluator. Both operands are already known to be numeric columns
/// (a general column falls back to scalar in the caller). Preserves the
/// division/modulo-by-zero fault scan and NaN-avoiding validity mask.
fn arith_vec_step(ctx: &Ctx, op: ArithOp, l: VVec, r: VVec, n: usize) -> VVec {
    let (ld, lv) = l.into_num();
    let (rd, rv) = r.into_num();
    if matches!(op, ArithOp::Div | ArithOp::Mod) {
        for i in 0..n {
            // Fault only when BOTH operands are non-null (`lv[i] && rv[i]`), matching
            // scalar `arith_step`: a null dividend short-circuits to null BEFORE the
            // divide-by-zero check, so `null / 0` is null (no fault), not an error.
            // The scalar path returns null here without faulting; the vectorized scan
            // omitted `lv[i]` and so faulted `null / 0`, diverging from TS/scalar.
            if lv[i] && rv[i] && rd[i] == 0.0 {
                ctx.set_fault(FAULT_DIV_ZERO);
                break;
            }
        }
    }
    let mut d = Vec::with_capacity(n);
    for i in 0..n {
        d.push(match op {
            ArithOp::Add => ld[i] + rd[i],
            ArithOp::Sub => ld[i] - rd[i],
            ArithOp::Mul => ld[i] * rd[i],
            ArithOp::Div => ld[i] / rd[i],
            ArithOp::Mod => ld[i] % rd[i],
        });
    }
    let valid = (0..n).map(|i| lv[i] && rv[i]).collect();
    VVec::Num { d, valid }
}

fn eval(env: &Env, expr: &CExpr) -> Val {
    match expr {
        CExpr::Lit(l) => match l {
            Lit::Null => Val::Null,
            Lit::Bool(b) => Val::Bool(*b),
            Lit::Num(n) => Val::Num(*n),
            Lit::Str(s) => vstr(s.as_str()),
            Lit::Temporal(t) => Val::Temporal(*t),
        },
        CExpr::Var(slot) => env.binding.get(*slot).cloned().unwrap_or(Val::Null),
        CExpr::Param(slot) => env.ctx.params.get(*slot).cloned().unwrap_or(Val::Null),
        CExpr::Prop { var_slot, key_ref } => {
            let bound = env.binding.get(*var_slot).cloned().unwrap_or(Val::Null);
            prop_of(env.graph, env.ctx, &bound, *key_ref)
        }
        CExpr::PropertyExists { var_slot, key_ref } => {
            let bound = env.binding.get(*var_slot).cloned().unwrap_or(Val::Null);
            prop_present(env.graph, env.ctx, &bound, *key_ref)
        }
        CExpr::List(items) => Val::List(items.iter().map(|e| eval(env, e)).collect()),
        // ISO record constructor → a canonical `Val::Record`: fields inserted in
        // sorted-key order, a duplicate field name taking the last value.
        CExpr::Record(fields) => {
            let mut out: Vec<(Arc<str>, Val)> = Vec::with_capacity(fields.len());
            for (k, e) in fields {
                let v = eval(env, e);
                match out.binary_search_by(|(ek, _)| ek.as_ref().cmp(k.as_ref())) {
                    Ok(i) => out[i].1 = v,
                    Err(i) => out.insert(i, (k.clone(), v)),
                }
            }
            Val::Record(out.into())
        }
        CExpr::Index { base, index } => {
            // ISO GQL list subscript `base[index]`: 0-based, out of range → null,
            // null-safe. A STRING index on a record/map is field access; a
            // non-string index / non-integer list index → null.
            let base_v = eval(env, base);
            let idx_v = eval(env, index);
            match base_v {
                // Match `Val::Num` directly — a non-number index (string, bool, …)
                // is null, NOT coerced via num_of. Both engines' contract is
                // "non-integer list index → null"; `num_of` silently turned `['1']`
                // into index 1 (and TS threw), so the two disagreed.
                Val::List(items) => match idx_v {
                    Val::Num(i) if i >= 0.0 && i.fract() == 0.0 && (i as usize) < items.len() => {
                        items[i as usize].clone()
                    }
                    _ => Val::Null,
                },
                Val::Record(pairs) => match &idx_v {
                    Val::Str(k) => map_get(&pairs, k),
                    _ => Val::Null,
                },
                _ => Val::Null,
            }
        }
        CExpr::Field {
            base,
            key_ref,
            name,
        } => {
            // `.field` chained off any expression. A record/map base reads the
            // field by name; a Node/Edge base reads the stored property via
            // `prop_of`; anything else → null, matching the bare `Prop` path.
            let base_v = eval(env, base);
            if let Val::Record(pairs) = &base_v {
                map_get(pairs, name)
            } else {
                prop_of(env.graph, env.ctx, &base_v, *key_ref)
            }
        }
        CExpr::PropField {
            var_slot,
            root_key_ref,
            descent,
        } => prop_field_of(env, *var_slot, *root_key_ref, descent),
        CExpr::Neg(e) => match arith_num(&eval(env, e), env.ctx) {
            Some(n) => Val::Num(-n),
            None => Val::Null,
        },
        // n-ary left-associative fold: `head` then each `(op, operand)` left to
        // right. Every operand is evaluated (no short-circuit — a fault in any
        // element still faults), matching the old left-nested binary tree.
        CExpr::Arith { head, tail } => {
            let mut acc = eval(env, head);
            for (op, e) in tail {
                let rv = eval(env, e);
                acc = arith_step(env, *op, acc, rv);
            }
            acc
        }
        CExpr::Concat(items) => {
            let mut acc = eval(env, &items[0]);
            for e in &items[1..] {
                let rv = eval(env, e);
                acc = concat_step(env, acc, rv);
            }
            acc
        }
        CExpr::Not(e) => truth_to_val(not3(as_truth(&eval(env, e)))),
        CExpr::And(items) => {
            let mut acc = as_truth(&eval(env, &items[0]));
            for e in &items[1..] {
                acc = and3(acc, as_truth(&eval(env, e)));
            }
            truth_to_val(acc)
        }
        CExpr::Or(items) => {
            let mut acc = as_truth(&eval(env, &items[0]));
            for e in &items[1..] {
                acc = or3(acc, as_truth(&eval(env, e)));
            }
            truth_to_val(acc)
        }
        CExpr::Xor(items) => {
            let mut acc = as_truth(&eval(env, &items[0]));
            for e in &items[1..] {
                acc = xor3(acc, as_truth(&eval(env, e)));
            }
            truth_to_val(acc)
        }
        CExpr::IsNull { expr, negated } => {
            let isnull = is_nullish(&eval(env, expr));
            Val::Bool(if *negated { !isnull } else { isnull })
        }
        CExpr::IsTruth {
            expr,
            truth,
            negated,
        } => {
            let m = as_truth(&eval(env, expr)) == *truth;
            Val::Bool(if *negated { !m } else { m })
        }
        CExpr::IsLabeled {
            expr,
            label,
            negated,
        } => {
            let el = eval(env, expr);
            let has = labels_match(env.graph, env.ctx, &el, label);
            Val::Bool(if *negated { !has } else { has })
        }
        CExpr::IsTyped {
            expr,
            ty,
            not_null,
            negated,
        } => {
            let m = value_is_typed_ty(&eval(env, expr), ty, *not_null);
            Val::Bool(if *negated { !m } else { m })
        }
        CExpr::GraphPred {
            kind,
            args,
            negated,
        } => eval_graph_pred(env, *kind, args, *negated),
        CExpr::In {
            expr,
            list,
            negated,
        } => {
            let r = in_list(&eval(env, expr), &eval(env, list));
            truth_to_val(if *negated { not3(r) } else { r })
        }
        CExpr::Compare { op, left, right } => {
            let lv = eval(env, left);
            let rv = eval(env, right);
            if is_nullish(&lv) || is_nullish(&rv) {
                return Val::Null; // UNKNOWN
            }
            compare_vals(env.ctx, *op, &lv, &rv)
        }
        CExpr::Case {
            subject,
            whens,
            else_,
        } => {
            if let Some(subj) = subject {
                let s = eval(env, subj);
                for (w, t) in whens {
                    let wv = eval(env, w);
                    if !is_nullish(&s) && !is_nullish(&wv) && val_eq(&s, &wv) {
                        return eval(env, t);
                    }
                }
            } else {
                for (w, t) in whens {
                    if as_truth(&eval(env, w)) == Some(true) {
                        return eval(env, t);
                    }
                }
            }
            else_.as_ref().map(|e| eval(env, e)).unwrap_or(Val::Null)
        }
        CExpr::Exists {
            patterns,
            where_,
            sub_len,
        } => Val::Bool(any_match(
            env.graph,
            env.ctx,
            patterns,
            where_.as_deref(),
            env.binding,
            *sub_len,
        )),
        CExpr::CountSubquery {
            patterns,
            where_,
            sub_len,
        } => Val::Num(count_matches(
            env.graph,
            env.ctx,
            patterns,
            where_.as_deref(),
            env.binding,
            *sub_len,
        ) as f64),
        CExpr::ValueSubquery {
            patterns,
            where_,
            ret,
            is_agg,
            sub_len,
        } => value_subquery(
            env.graph,
            env.ctx,
            SubqueryPlan {
                patterns,
                where_: where_.as_deref(),
                ret,
                is_agg: *is_agg,
                sub_len: *sub_len,
            },
            env.binding,
        ),
        CExpr::LetIn { bindings, body } => {
            // Bind each local into a per-eval clone (left-to-right, so a later
            // binding sees earlier ones), then evaluate the body against it. The
            // group / aggregate context is preserved so an aggregate binding folds
            // over the same group and the body reads the resulting scalar.
            let mut local = env.binding.clone();
            for (slot, cexpr) in bindings {
                let v = {
                    let e = Env {
                        binding: &local,
                        ..*env
                    };
                    eval(&e, cexpr)
                };
                local.set(*slot, v);
            }
            let e = Env {
                binding: &local,
                ..*env
            };
            eval(&e, body)
        }
        CExpr::Scalar { func, args } => {
            if matches!(func, ScalarFn::Unknown) {
                env.ctx.set_fault(FAULT_UNKNOWN_FN); // fail loud, not silent NULL
            }
            let vals: Vec<Val> = args.iter().map(|a| eval(env, a)).collect();
            call_scalar(env.graph, env.ctx, *func, &vals)
        }
        CExpr::Aggregate {
            func,
            arg,
            distinct,
            star,
            frac,
        } => eval_aggregate(env, *func, arg.as_deref(), *distinct, *star, *frac),
        CExpr::AggRef(idx) => env
            .agg_values
            .and_then(|a| a.get(*idx))
            .cloned()
            .unwrap_or(Val::Null),
    }
}

thread_local! {
    /// Reusable operand stack for the expression VM. The VM is never re-entrant
    /// on its own stack (the only recursion is `Op::Tree`, which calls the
    /// tree-walking `eval`, not `run`), so a single per-thread buffer is safe and
    /// keeps the hot per-row path allocation-free.
    static VM_STACK: std::cell::RefCell<Vec<Val>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Evaluate a projection item, routing through the VM or tree-walk per [`USE_VM`].
#[inline]
fn eval_item(env: &Env, item: &super::plan::CReturnItem) -> Val {
    if USE_VM {
        run(env, &item.prog)
    } else {
        eval(env, &item.expr)
    }
}

/// Execute a compiled expression [`Program`] (stack machine) against `env`.
/// Mirrors [`eval`] op-for-op; `Op::Tree` delegates the un-compilable
/// subexpressions (CASE / EXISTS / COUNT{} / aggregate) back to `eval`.
fn run(env: &Env, prog: &Program) -> Val {
    VM_STACK.with(|cell| {
        let mut st = cell.borrow_mut();
        let base = st.len();
        for op in &prog.0 {
            match op {
                Op::Const(l) => st.push(match l {
                    Lit::Null => Val::Null,
                    Lit::Bool(b) => Val::Bool(*b),
                    Lit::Num(n) => Val::Num(*n),
                    Lit::Str(s) => vstr(s.as_str()),
                    Lit::Temporal(t) => Val::Temporal(*t),
                }),
                Op::Var(slot) => st.push(env.binding.get(*slot).cloned().unwrap_or(Val::Null)),
                Op::Param(slot) => st.push(env.ctx.params.get(*slot).cloned().unwrap_or(Val::Null)),
                Op::Prop { var_slot, key_ref } => {
                    let bound = env.binding.get(*var_slot).cloned().unwrap_or(Val::Null);
                    st.push(prop_of(env.graph, env.ctx, &bound, *key_ref));
                }
                Op::MakeList(n) => {
                    let at = st.len() - n;
                    let items = st.split_off(at);
                    st.push(Val::list(items));
                }
                Op::Arith(op) => {
                    let bv = st.pop().unwrap();
                    let av = st.pop().unwrap();
                    let out = if matches!(av, Val::Temporal(_)) || matches!(bv, Val::Temporal(_)) {
                        temporal_arith(env.ctx, *op, &av, &bv)
                    } else {
                        match (arith_num(&av, env.ctx), arith_num(&bv, env.ctx)) {
                            (Some(a), Some(b)) => {
                                if matches!(op, ArithOp::Div | ArithOp::Mod) && b == 0.0 {
                                    env.ctx.set_fault(FAULT_DIV_ZERO);
                                    Val::Null
                                } else {
                                    Val::Num(match op {
                                        ArithOp::Add => a + b,
                                        ArithOp::Sub => a - b,
                                        ArithOp::Mul => a * b,
                                        ArithOp::Div => a / b,
                                        ArithOp::Mod => a % b,
                                    })
                                }
                            }
                            _ => Val::Null,
                        }
                    };
                    st.push(out);
                }
                Op::Compare(op) => {
                    let rv = st.pop().unwrap();
                    let lv = st.pop().unwrap();
                    st.push(if is_nullish(&lv) || is_nullish(&rv) {
                        Val::Null
                    } else {
                        compare_vals(env.ctx, *op, &lv, &rv)
                    });
                }
                Op::Concat => {
                    let rv = st.pop().unwrap();
                    let lv = st.pop().unwrap();
                    st.push(match (&lv, &rv) {
                        _ if is_nullish(&lv) || is_nullish(&rv) => Val::Null,
                        // ISO GQL `||`: list ++ list concatenates the two lists;
                        // otherwise it is string concatenation (unchanged).
                        (Val::List(a), Val::List(b)) => {
                            Val::List(a.iter().chain(b.iter()).cloned().collect())
                        }
                        _ => vstr(js_str(env.graph, &lv) + &js_str(env.graph, &rv)),
                    });
                }
                Op::Neg => {
                    let v = st.pop().unwrap();
                    st.push(match arith_num(&v, env.ctx) {
                        Some(n) => Val::Num(-n),
                        None => Val::Null,
                    });
                }
                Op::Not => {
                    let v = st.pop().unwrap();
                    st.push(truth_to_val(not3(as_truth(&v))));
                }
                Op::And => {
                    let b = as_truth(&st.pop().unwrap());
                    let a = as_truth(&st.pop().unwrap());
                    st.push(truth_to_val(and3(a, b)));
                }
                Op::Or => {
                    let b = as_truth(&st.pop().unwrap());
                    let a = as_truth(&st.pop().unwrap());
                    st.push(truth_to_val(or3(a, b)));
                }
                Op::Xor => {
                    let b = as_truth(&st.pop().unwrap());
                    let a = as_truth(&st.pop().unwrap());
                    st.push(truth_to_val(xor3(a, b)));
                }
                Op::IsNull(negated) => {
                    let isnull = is_nullish(&st.pop().unwrap());
                    st.push(Val::Bool(if *negated { !isnull } else { isnull }));
                }
                Op::IsTruth(truth, negated) => {
                    let m = as_truth(&st.pop().unwrap()) == *truth;
                    st.push(Val::Bool(if *negated { !m } else { m }));
                }
                Op::IsLabeled(label, negated) => {
                    let el = st.pop().unwrap();
                    let has = labels_match(env.graph, env.ctx, &el, label);
                    st.push(Val::Bool(if *negated { !has } else { has }));
                }
                Op::In(negated) => {
                    let list = st.pop().unwrap();
                    let expr = st.pop().unwrap();
                    let r = in_list(&expr, &list);
                    st.push(truth_to_val(if *negated { not3(r) } else { r }));
                }
                Op::Scalar(func, argc) => {
                    if matches!(func, ScalarFn::Unknown) {
                        env.ctx.set_fault(FAULT_UNKNOWN_FN);
                    }
                    let at = st.len() - argc;
                    let args = st.split_off(at);
                    st.push(call_scalar(env.graph, env.ctx, *func, &args));
                }
                Op::AggRef(idx) => {
                    st.push(
                        env.agg_values
                            .and_then(|a| a.get(*idx))
                            .cloned()
                            .unwrap_or(Val::Null),
                    );
                }
                Op::Tree(e) => {
                    let v = eval(env, e);
                    st.push(v);
                }
            }
        }
        // The program leaves exactly one value above `base`.
        debug_assert_eq!(st.len(), base + 1);
        st.pop().unwrap_or(Val::Null)
    })
}

fn eval_aggregate(
    env: &Env,
    func: AggFn,
    arg: Option<&CExpr>,
    distinct: bool,
    star: bool,
    frac: Option<f64>,
) -> Val {
    let single;
    let group: &[Binding] = match env.group {
        Some(g) => g,
        None => {
            single = [env.binding.clone()];
            &single
        }
    };
    if func == AggFn::Count && star {
        return Val::Num(group.len() as f64);
    }
    let Some(arg) = arg else { return Val::Null };
    // Evaluate the argument over every binding in the group.
    let raw: Vec<Val> = group
        .iter()
        .map(|b| {
            let e = Env {
                graph: env.graph,
                ctx: env.ctx,
                binding: b,
                group: Some(group),
                agg_values: None,
            };
            eval(&e, arg)
        })
        .collect();
    let mut values: Vec<Val> = raw.into_iter().filter(|v| !is_nullish(v)).collect();
    if distinct {
        // Mirror Agg::step: dedup element values by dense id, scalars by `val_key`.
        let mut seen: FxHashSet<String> = FxHashSet::default();
        let mut seen_ids: FxHashSet<u64> = FxHashSet::default();
        values.retain(|v| match v {
            Val::Node(i) => seen_ids.insert(*i as u64),
            Val::Edge(i) => seen_ids.insert(*i as u64 | EDGE_ID_TAG),
            _ => {
                let mut k = String::new();
                val_key(v, &mut k);
                seen.insert(k)
            }
        });
    }
    // Determine the group's aggregate type by scanning ALL values, not just the
    // first — order-independent, matching the TS engine. Checking only `first()`
    // made `sum([5, DATE])` take the numeric arm (coercing DATE → NaN → null) while
    // TS threw, and made the result depend on scan order.
    let has_temporal = values.iter().any(|v| matches!(v, Val::Temporal(_)));
    let has_list = values.iter().any(|v| matches!(v, Val::List(_)));
    match func {
        AggFn::Count => Val::Num(values.len() as f64),
        // `sum` over DURATIONs computes; any non-DURATION temporal (or a numeric
        // mixed in) faults via `temporal_values_sum`.
        AggFn::Sum if has_temporal => temporal_values_sum(&values, env.ctx),
        // A list/map isn't summable — fault loud (matches TS `nonNumericAgg`) rather
        // than coercing it to NaN → null.
        AggFn::Sum if has_list => {
            env.ctx.set_fault(FAULT_NONNUMERIC_AGG);
            Val::Null
        }
        AggFn::Sum => Val::Num(values.iter().filter_map(num_of).sum()),
        // `avg` over any temporal faults (needs unrepresentable duration÷count).
        AggFn::Avg if has_temporal => {
            env.ctx.set_fault(FAULT_TEMPORAL_AGG);
            Val::Null
        }
        AggFn::Avg if has_list => {
            env.ctx.set_fault(FAULT_NONNUMERIC_AGG);
            Val::Null
        }
        AggFn::Avg => {
            if values.is_empty() {
                Val::Null
            } else {
                let s: f64 = values.iter().filter_map(num_of).sum();
                Val::Num(s / values.len() as f64)
            }
        }
        AggFn::Min => fold_extreme(values, Ordering::Less),
        AggFn::Max => fold_extreme(values, Ordering::Greater),
        AggFn::CollectList => Val::list(values),
        AggFn::PercentileCont => percentile(&values, frac.unwrap_or(0.0), true),
        AggFn::PercentileDisc => percentile(&values, frac.unwrap_or(0.0), false),
        AggFn::StddevPop | AggFn::StddevSamp => {
            let (mut n, mut sum, mut sum_sq) = (0u64, 0.0f64, 0.0f64);
            for x in values.iter().filter_map(num_of) {
                sum += x;
                sum_sq += x * x;
                n += 1;
            }
            stddev_of(n, sum, sum_sq, func == AggFn::StddevSamp)
        }
    }
}

/// Population / sample standard deviation from the one-pass moments. `stddev_pop`
/// is null over 0 rows (else `sqrt(Σx²/n − mean²)`); `stddev_samp` is null over
/// fewer than 2 rows (dividing the summed squared deviations by `n−1`). The summed
/// squared deviation is clamped at 0 so floating-point cancellation can't make a
/// tiny negative slip into `sqrt` (→ NaN). Written identically in both engines.
fn stddev_of(n: u64, sum: f64, sum_sq: f64, sample: bool) -> Val {
    let denom = if sample {
        if n < 2 {
            return Val::Null;
        }
        (n - 1) as f64
    } else {
        if n == 0 {
            return Val::Null;
        }
        n as f64
    };
    let nf = n as f64;
    let variance = (sum_sq - sum * sum / nf) / denom;
    // Clamp a negative variance to 0 WITHOUT swallowing NaN. `f64::max` returns
    // the non-NaN operand — `f64::NAN.max(0.0)` is 0.0 — while the TS twin's
    // `Math.max(0, NaN)` is NaN. A non-numeric value in the group (`stddev_pop`
    // over a string column) makes the variance NaN, which has to stay NaN (→
    // null, like `avg`) rather than render as a real 0.
    let clamped = if variance.is_nan() {
        f64::NAN
    } else {
        variance.max(0.0)
    };
    Val::Num(clamped.sqrt())
}

/// ISO ordered-set percentile over a group's numeric values. `cont` (=
/// `percentile_cont`) interpolates linearly between the two ranks bracketing
/// `frac·(n−1)`; otherwise (`percentile_disc`) it returns the value at the smallest
/// 0-based rank `k` with `(k+1)/n ≥ frac`. Non-numeric / non-finite values are
/// dropped; `frac` is pre-clamped to `[0, 1]`. Empty input → `Null`.
fn percentile(values: &[Val], frac: f64, cont: bool) -> Val {
    let mut nums: Vec<f64> = values
        .iter()
        .filter_map(num_of)
        .filter(|x| x.is_finite())
        .collect();
    if nums.is_empty() {
        return Val::Null;
    }
    nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
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
    Val::Num(result)
}

fn fold_extreme(values: Vec<Val>, want: Ordering) -> Val {
    let mut it = values.into_iter();
    let Some(mut acc) = it.next() else {
        return Val::Null;
    };
    for v in it {
        if cmp_total(&v, &acc) == want {
            acc = v;
        }
    }
    acc
}

// --- scalar functions (dispatched on the resolved enum) ---
mod scalar_fns;
use scalar_fns::*;

// --- pattern matching --------------------------------------------------------

/// Bind a slot to `value` for a recursion branch, returning whether it was newly
/// set (so the caller can restore it on backtrack). A consistent already-bound
/// slot is left untouched; an inconsistent one is rejected (`None`).
fn bind_slot(binding: &mut Binding, slot: Option<usize>, value: &Val) -> Option<bool> {
    match slot {
        None => Some(false),
        Some(s) => {
            if binding.bound(s) {
                if val_eq(binding.get(s).unwrap(), value) {
                    Some(false)
                } else {
                    None // join conflict — this branch fails
                }
            } else {
                binding.set(s, value.clone());
                Some(true)
            }
        }
    }
}

fn satisfies(
    graph: &Graph,
    ctx: &Ctx,
    element: &Val,
    props: &[CPropConstraint],
    where_: Option<&CExpr>,
    binding: &Binding,
) -> bool {
    let env = Env::new(graph, ctx, binding);
    for pc in props {
        if !val_eq(
            &prop_of(graph, ctx, element, pc.key_ref),
            &eval(&env, &pc.value),
        ) {
            return false;
        }
    }
    where_.is_none_or(|w| as_truth(&eval(&env, w)) == Some(true))
}

/// A label this expression *guarantees* (for seeding from a label bucket): the
/// ref of a bare label or a conjunct; `or`/`not`/`%` can't narrow.
fn seed_label(expr: &CLabelExpr) -> Option<usize> {
    match expr {
        CLabelExpr::Label(r) => Some(*r),
        CLabelExpr::And(l, r) => seed_label(l).or_else(|| seed_label(r)),
        _ => None,
    }
}

/// Run `f` over each seed vertex, returning `false` if `f` requested a stop.
/// Iterates the label bucket / live-vertex range directly — no `Vec` of seeds.
fn for_each_seed(
    graph: &Graph,
    ctx: &Ctx,
    label: Option<&CLabelExpr>,
    f: &mut dyn FnMut(u32) -> bool,
) -> bool {
    match label.and_then(seed_label) {
        Some(r) => match ctx.labels[r].0 {
            Some(lid) => graph.vertices_with_label(lid).iter().all(|&s| f(s)),
            None => true, // unknown label → no seeds
        },
        None => graph.vertex_indices().all(f),
    }
}

/// Expand one segment from `v` as `(edge index, neighbor)` — a lazy iterator
/// (no intermediate `Vec`), so a short-circuiting consumer stops walking early.
fn expand<'a>(
    graph: &'a Graph,
    ctx: &'a Ctx,
    v: u32,
    direction: Direction,
    label: Option<&'a CLabelExpr>,
) -> impl Iterator<Item = (u32, u32)> + 'a {
    let out = matches!(direction, Direction::Out | Direction::Both).then(|| graph.out_adj(v));
    let inn = matches!(direction, Direction::In | Direction::Both).then(|| graph.in_adj(v));
    // A self-loop sits in both the out- and in-index of `v`, so an undirected
    // (`Both`) walk would yield it twice — once per side. The out-side already
    // emits it; drop it from the in-side (`a.nbr == v` ⇔ the far end is also `v`,
    // i.e. a self-loop). Directed In/Out keep it. The `!both` guard short-circuits
    // so directed traversal pays nothing.
    let both = matches!(direction, Direction::Both);
    out.into_iter()
        .flatten()
        .chain(
            inn.into_iter()
                .flatten()
                .filter(move |a| !both || a.nbr != v),
        )
        .filter(move |a| label.is_none_or(|e| eval_label_edge(ctx, a.etype, e)))
        .map(|a| (a.eidx, a.nbr))
}

/// Try to match `node` at vertex `vi`, extending `binding` in place and invoking
/// `cont` on success, then restoring it. Returns `false` only if `cont` asked to
/// stop the whole traversal.
fn match_node_then<C: FnMut(&mut Binding) -> bool + ?Sized>(
    graph: &Graph,
    ctx: &Ctx,
    binding: &mut Binding,
    node: &CNode,
    vi: u32,
    cont: &mut C,
) -> bool {
    if !matches_label(graph, ctx, vi, node.label.as_ref()) {
        return true; // no match here, but keep going
    }
    let Some(did_set) = bind_slot(binding, node.var_slot, &Val::Node(vi)) else {
        return true; // join conflict
    };
    let go = satisfies(
        graph,
        ctx,
        &Val::Node(vi),
        &node.props,
        node.where_.as_ref(),
        binding,
    );
    let keep = if go { cont(binding) } else { true };
    if did_set {
        binding.unset(node.var_slot.unwrap());
    }
    keep
}

// Pattern traversal / walks / shortest-path / bidirectional search.
mod pathfind;
use pathfind::*;

// Match execution: drive matches, aggregate, project to rows.
mod matcher;
use matcher::*;

// --- vectorized (batched) node scan -----------------------------------------
//
// One operation across the whole matched row set instead of per row. A column
// of evaluated values; numeric data is a flat `Vec<f64>` with a validity mask
// so arithmetic/comparison loops stay branch-light and autovectorizable. Three
// representations: numeric, three-valued boolean, and a `Gen` escape hatch for
// anything outside the numeric subset (strings, CASE, identity, subqueries),
// evaluated per row by the scalar `eval` for just that column.
enum VVec {
    Num { d: Vec<f64>, valid: Vec<bool> },
    Bool { t: Vec<bool>, valid: Vec<bool> },
    Gen(Vec<Val>),
}

impl VVec {
    /// Coerce to numeric (`num_of` semantics): invalid where the source is null.
    fn into_num(self) -> (Vec<f64>, Vec<bool>) {
        match self {
            Self::Num { d, valid } => (d, valid),
            Self::Bool { t, valid } => (
                t.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
                valid,
            ),
            Self::Gen(vs) => {
                let mut d = Vec::with_capacity(vs.len());
                let mut valid = Vec::with_capacity(vs.len());
                for v in &vs {
                    match num_of(v) {
                        Some(x) => {
                            d.push(x);
                            valid.push(true);
                        }
                        None => {
                            d.push(f64::NAN);
                            valid.push(false);
                        }
                    }
                }
                (d, valid)
            }
        }
    }

    /// Per-row Kleene truth (for WHERE and boolean connectives).
    fn into_truth(self) -> Vec<Truth> {
        match self {
            Self::Bool { t, valid } => t
                .iter()
                .zip(&valid)
                .map(|(&b, &v)| v.then_some(b))
                .collect(),
            Self::Num { d, valid } => d
                .iter()
                .zip(&valid)
                .map(|(&x, &v)| v.then_some(x != 0.0 && !x.is_nan()))
                .collect(),
            Self::Gen(vs) => vs.iter().map(as_truth).collect(),
        }
    }

    /// Convert directly to a typed Arrow column — `Num`/`Bool` move their `f64`/
    /// `bool` buffers in with no `Val` boxing; `Gen` flattens its `Val`s (elements
    /// → ids) and infers the physical type. This is the boxing-free result path.
    #[cfg(feature = "arrow")]
    fn into_arrow(self, graph: &Graph) -> ArrowColumn {
        let opt = |v: Vec<bool>| if v.iter().all(|&b| b) { None } else { Some(v) };
        match self {
            Self::Num { d, valid } => ArrowColumn::Num {
                data: d,
                valid: opt(valid),
            },
            Self::Bool { t, valid } => ArrowColumn::Bool {
                data: t,
                valid: opt(valid),
            },
            Self::Gen(vals) => {
                let values: Vec<Value> = vals.iter().map(|v| val_to_value(graph, v)).collect();
                ArrowColumn::from_values(values.iter())
            }
        }
    }

    /// Keep only rows `[start, end)` (for SKIP/LIMIT on a typed column). Only the
    /// Arrow fast path slices typed columns; the RowSet path slices `ScanCols`.
    #[cfg(feature = "arrow")]
    fn slice(self, start: usize, end: usize) -> Self {
        match self {
            Self::Num { d, valid } => Self::Num {
                d: d[start..end].to_vec(),
                valid: valid[start..end].to_vec(),
            },
            Self::Bool { t, valid } => Self::Bool {
                t: t[start..end].to_vec(),
                valid: valid[start..end].to_vec(),
            },
            Self::Gen(v) => Self::Gen(v[start..end].to_vec()),
        }
    }

    /// The `i`-th value as a core output [`Value`], read straight from the typed
    /// buffer — a numeric/bool column skips `Val` boxing entirely (`f64` →
    /// `Value::Num`); a `Gen` column converts its per-row `Val`. Used by the fused
    /// terminal transpose to avoid materializing an intermediate `Vec<Val>` column.
    fn value_at(&self, i: usize, graph: &Graph) -> Value {
        match self {
            Self::Num { d, valid } => {
                if valid[i] {
                    Value::Num(d[i])
                } else {
                    Value::Null
                }
            }
            Self::Bool { t, valid } => {
                if valid[i] {
                    Value::Bool(t[i])
                } else {
                    Value::Null
                }
            }
            Self::Gen(vs) => val_to_value(graph, &vs[i]),
        }
    }

    /// Final per-row output values (for projection cells).
    fn into_vals(self) -> Vec<Val> {
        match self {
            Self::Num { d, valid } => d
                .iter()
                .zip(&valid)
                .map(|(&x, &v)| if v { Val::Num(x) } else { Val::Null })
                .collect(),
            Self::Bool { t, valid } => t
                .iter()
                .zip(&valid)
                .map(|(&b, &v)| if v { Val::Bool(b) } else { Val::Null })
                .collect(),
            Self::Gen(vs) => vs,
        }
    }
}

/// A unary math `ScalarFn` over f64, if `func` is one (so it vectorizes).
fn unary_math(func: ScalarFn) -> Option<fn(f64) -> f64> {
    use ScalarFn::*;
    Some(match func {
        Abs => f64::abs,
        Ceil => f64::ceil,
        Floor => f64::floor,
        Sqrt => f64::sqrt,
        Exp => f64::exp,
        Ln => f64::ln,
        Log10 => f64::log10,
        Sin => f64::sin,
        Cos => f64::cos,
        Tan => f64::tan,
        Asin => f64::asin,
        Acos => f64::acos,
        Atan => f64::atan,
        Sinh => f64::sinh,
        Cosh => f64::cosh,
        Tanh => f64::tanh,
        // Same multiply-then-divide association as the scalar arm in
        // `scalar_fns.rs` — `to_degrees`/`to_radians` pre-round the constant and
        // land one ulp off. This vectorized table is a SECOND dispatch site, so a
        // fix to one is only half a fix.
        Degrees => |n| (n * 180.0) / std::f64::consts::PI,
        Radians => |n| (n * std::f64::consts::PI) / 180.0,
        _ => return None,
    })
}

/// Which element kind a scanned binding slot holds (so a `Prop` reads the right
/// property store — vertex vs edge — at that slot's per-row ids).
#[derive(Clone, Copy, PartialEq)]
enum Elem {
    Node,
    Edge,
}

/// The matched row set as parallel columns. Each binding slot is either an
/// *element* column (kind + per-row dense id — fast to gather props from, and the
/// only thing a traversal can expand) or, once a `WITH` projects computed values,
/// a *value* column (per-row `Val`). A "row" is one full match. This is what
/// every vectorized expression reads from, so traversals (`a`, `r`, `b` slots)
/// and a single-node scan (`n` slot) look the same — just more slots — and a
/// pipeline `WITH` can carry elements forward as fast columns while adding
/// computed value columns beside them.
struct ScanCols {
    n: usize,
    slots: Vec<Option<(Elem, Vec<u32>)>>,
    /// Computed value columns, parallel to `slots` (set only post-projection).
    vals: Vec<Option<Vec<Val>>>,
}

impl ScanCols {
    fn new(scope_len: usize) -> Self {
        let w = scope_len.max(1);
        Self {
            n: 0,
            slots: (0..w).map(|_| None).collect(),
            vals: (0..w).map(|_| None).collect(),
        }
    }
    fn slot(&self, s: usize) -> Option<(Elem, &[u32])> {
        self.slots
            .get(s)
            .and_then(|o| o.as_ref())
            .map(|(e, v)| (*e, v.as_slice()))
    }
    fn val_slot(&self, s: usize) -> Option<&[Val]> {
        self.vals.get(s).and_then(|o| o.as_deref())
    }
}

/// Gather a numeric `Column` at `ids` into a `VVec::Num` (+ validity mask), or
/// `None` if the column isn't numeric (caller then falls back to per-row `Gen`).
fn gather_num(col: Option<&Column>, ids: &[u32]) -> Option<VVec> {
    match col {
        Some(Column::Num { data, present }) => {
            let mut d = Vec::with_capacity(ids.len());
            let mut valid = Vec::with_capacity(ids.len());
            for &vi in ids {
                let i = vi as usize;
                d.push(data[i]);
                valid.push(present.get(i));
            }
            Some(VVec::Num { d, valid })
        }
        _ => None,
    }
}

/// Gather a `Column::Str` at `ids` into a `VVec::Gen` of `Val::Str` (shared Arc
/// clones; absent → `Null`) — the string analogue of [`gather_num`]. Replaces the
/// per-row `Binding` rebuild + `eval` dispatch of `scalar_col` with a tight
/// interner-clone loop, so projecting/sorting a string column stays cheap.
fn gather_str(col: Option<&Column>, ids: &[u32], strs: &crate::graph::Dict) -> Option<VVec> {
    match col {
        Some(Column::Str { data, present }) => Some(VVec::Gen(
            ids.iter()
                .map(|&vi| {
                    let i = vi as usize;
                    if present.get(i) {
                        Val::Str(strs.arc(data[i]))
                    } else {
                        Val::Null
                    }
                })
                .collect(),
        )),
        _ => None,
    }
}

/// Gather a `Column::Temporal` at `ids` into a `VVec::Gen` of `Val::Temporal`
/// (absent → `Null`) — the temporal analogue of [`gather_str`]. Reconstructs each
/// value straight from the packed per-type arrays in a tight loop, replacing the
/// per-row `Binding` rebuild + `eval` dispatch of `scalar_col` — so projecting or
/// ordering a temporal column engages the vectorized scan instead of falling back.
fn gather_temporal(col: Option<&Column>, ids: &[u32]) -> Option<VVec> {
    match col {
        Some(Column::Temporal { data, present }) => Some(VVec::Gen(
            ids.iter()
                .map(|&vi| {
                    let i = vi as usize;
                    if present.get(i) {
                        Val::Temporal(data.get(i))
                    } else {
                        Val::Null
                    }
                })
                .collect(),
        )),
        _ => None,
    }
}

/// If `e` is a `Prop` over a typed `Column::Temporal` in `sc`, gather it as a
/// column of `Option<Temporal>` (absent → `None`) for a typed ORDER BY — the sort
/// then compares Copy temporals via `cmp_total`, skipping the `Val` wrapper +
/// dispatch of the generic `Vec<Val>` keycol. `None` (→ generic sort) otherwise.
fn temporal_sort_key(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    e: &CExpr,
) -> Option<Vec<Option<crate::temporal::Temporal>>> {
    let CExpr::Prop { var_slot, key_ref } = e else {
        return None;
    };
    let (elem, ids) = sc.slot(*var_slot)?;
    let (store, kid) = match elem {
        Elem::Node => (&graph.props, ctx.prop_keys[*key_ref].0),
        Elem::Edge => (&graph.edge_props, ctx.prop_keys[*key_ref].1),
    };
    let Some(Column::Temporal { data, present }) = kid.and_then(|k| store.cols.get(k as usize))
    else {
        return None;
    };
    Some(
        ids.iter()
            .map(|&vi| {
                let i = vi as usize;
                present.get(i).then(|| data.get(i))
            })
            .collect(),
    )
}

/// A single-key **dense** ORDER BY sort column: packed `i128` keys + a presence
/// flag per row, plus the key's `descending` and `nulls_first` spec.
type DenseSortCol = (Vec<i128>, Vec<bool>, bool, Option<bool>);
/// A single-key **typed** ORDER BY sort column (Duration): `Option<Temporal>` per
/// row (absent → `None`), plus `descending` and `nulls_first`.
type TypedSortCol = (Vec<Option<crate::temporal::Temporal>>, bool, Option<bool>);

/// If `e` is a `Prop` over an **instant** `Column::Temporal` in `sc` (every kind
/// except `Duration`), gather a **dense** sort key: one `i128` per row (packed by
/// [`TemporalCol::monotonic_key`]) + a presence flag. The key is monotonic with
/// `cmp_total` within the column, so it sorts cache-friendly like a numeric column
/// — no `Val`, no `cmp_total` dispatch — which is what makes `ORDER BY ts LIMIT n`
/// (top-k) fast. `None` for `Duration` (uses the `cmp_total` comparator) or a
/// non-temporal / mixed column (generic sort).
fn dense_sort_key(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    e: &CExpr,
) -> Option<(Vec<i128>, Vec<bool>)> {
    let CExpr::Prop { var_slot, key_ref } = e else {
        return None;
    };
    let (elem, ids) = sc.slot(*var_slot)?;
    let (store, kid) = match elem {
        Elem::Node => (&graph.props, ctx.prop_keys[*key_ref].0),
        Elem::Edge => (&graph.edge_props, ctx.prop_keys[*key_ref].1),
    };
    let Some(Column::Temporal { data, present }) = kid.and_then(|k| store.cols.get(k as usize))
    else {
        return None;
    };
    // Probe the first present row: a `Duration` column has no `i128` key → bail.
    data.monotonic_key(*ids.first()? as usize)?;
    let mut key = Vec::with_capacity(ids.len());
    let mut valid = Vec::with_capacity(ids.len());
    for &vi in ids {
        let i = vi as usize;
        key.push(data.monotonic_key(i).unwrap_or(0));
        valid.push(present.get(i));
    }
    Some((key, valid))
}

/// `compare_sort` for the dense `i128` key: byte-identical to [`compare_sort`]
/// (same null placement + direction) but on `(i128 key, present)` — a plain integer
/// compare, no `Val`, no `cmp_total` dispatch.
#[inline]
fn dense_compare_sort(
    a: i128,
    a_present: bool,
    b: i128,
    b_present: bool,
    descending: bool,
    nulls_first: Option<bool>,
) -> Ordering {
    match (a_present, b_present) {
        (false, false) => Ordering::Equal,
        (false, true) | (true, false) => {
            let first = nulls_first.unwrap_or(false);
            // `a` is the null iff `!a_present`; place it first when NULLS FIRST.
            if a_present != first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (true, true) => {
            let base = a.cmp(&b);
            if descending {
                base.reverse()
            } else {
                base
            }
        }
    }
}

/// `compare_sort` for a typed temporal key — byte-identical to the generic
/// [`compare_sort`] (both-null → Equal; one-null → absolute placement, default
/// last, independent of ASC/DESC; both-present → `cmp_total`, reversed if
/// descending) but on `Option<Temporal>` instead of `&Val`.
fn temporal_compare_sort(
    a: &Option<crate::temporal::Temporal>,
    b: &Option<crate::temporal::Temporal>,
    descending: bool,
    nulls_first: Option<bool>,
) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) | (Some(_), None) => {
            let first = nulls_first.unwrap_or(false);
            if a.is_none() == first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (Some(x), Some(y)) => {
            let base = x.cmp_total(y);
            if descending {
                base.reverse()
            } else {
                base
            }
        }
    }
}

/// Scalar fallback: evaluate `e` once per row into a `Vec<Val>` (the slow path
/// for any subexpression outside the numeric vector subset). Reuses one binding,
/// setting every scanned slot to its per-row element.
fn scalar_col(graph: &Graph, ctx: &Ctx, sc: &ScanCols, e: &CExpr) -> Vec<Val> {
    // Bind row `i`'s frame columns into `b`, then evaluate `e`. `Gen` columns are where
    // per-row subqueries (EXISTS / COUNT / pattern) and other non-vectorizable exprs land,
    // so this is often the heaviest per-row work in a projection or WHERE.
    let bind_and_eval = |b: &mut Binding, i: usize| -> Val {
        for (slot, col) in sc.slots.iter().enumerate() {
            if let Some((elem, ids)) = col {
                b.set(
                    slot,
                    match elem {
                        Elem::Node => Val::Node(ids[i]),
                        Elem::Edge => Val::Edge(ids[i]),
                    },
                );
            } else if let Some(vals) = &sc.vals[slot] {
                b.set(slot, vals[i].clone());
            }
        }
        eval(&Env::new(graph, ctx, b), e)
    };

    // Rows are independent, so for a large frame split them across rayon threads: each
    // thread reuses its own `Binding`, and an indexed collect preserves row order. Sound
    // because `&Ctx` is `Sync` under this feature (atomic fault + `Mutex` trail-mark pool).
    #[cfg(feature = "parallel-query")]
    {
        const MIN_ROWS: usize = 8_192;
        if sc.n >= MIN_ROWS && rayon::current_num_threads() > 1 {
            return (0..sc.n)
                .into_par_iter()
                .map_init(
                    || Binding(vec![None; sc.slots.len()]),
                    |b, i| bind_and_eval(b, i),
                )
                .collect();
        }
    }

    let mut b = Binding(vec![None; sc.slots.len()]);
    (0..sc.n).map(|i| bind_and_eval(&mut b, i)).collect()
}

/// Evaluate `e` over the whole matched row set `sc`. Numeric and boolean
/// subtrees stay vectorized; everything else degrades to a per-row `Gen` column.
/// Vectorized `Prop = / <> str-literal` over a `Column::Str`: compare the stored
/// dictionary **ids** (u32) to the literal's id — resolved once through the graph
/// interner — instead of byte-comparing an `Arc<str>` per row. A literal that
/// isn't in the interner equals no stored string (every stored string is
/// interned), so `=` is all-false and `<>` all-true with **zero** string bytes
/// read. Returns `None` unless exactly one side is a direct Str-column `Prop` and
/// the other a string literal, and the op is `Eq`/`Ne` (interner id order is
/// arbitrary, not lexicographic, so `<`/`>` can't use it). Absent property ⇒ the
/// row's result is UNKNOWN (invalid), matching three-valued logic.
fn str_eq_vec(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    op: CompareOp,
    left: &CExpr,
    right: &CExpr,
) -> Option<VVec> {
    if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
        return None;
    }
    // Match the (prop, string scalar) pair in either operand order. The scalar is
    // a literal OR a `$param` bound to a string: this used to accept only the
    // literal, so `WHERE u.name = $n` — the parameterized, injection-safe way to
    // write the SAME query — fell through to the per-row scalar path and cost 35x
    // (29.6 us vs 1038 us over 20k rows; ~52 ns/row, flat in graph size, so the
    // ratio only grows). `temporal_cmp_vec` below has always taken a param here;
    // the two recognizers simply drifted.
    let (prop, lit) = match (left, right) {
        (p @ CExpr::Prop { .. }, other) | (other, p @ CExpr::Prop { .. }) => {
            (p, scalar_str(other, ctx)?)
        }
        _ => return None,
    };
    let CExpr::Prop { var_slot, key_ref } = prop else {
        return None;
    };
    let (elem, ids) = sc.slot(*var_slot)?;
    let (store, kid) = match elem {
        Elem::Node => (&graph.props, ctx.prop_keys[*key_ref].0),
        Elem::Edge => (&graph.edge_props, ctx.prop_keys[*key_ref].1),
    };
    let Some(Column::Str { data, present }) = kid.and_then(|k| store.cols.get(k as usize)) else {
        return None; // not a typed string column (Mixed/absent) — scalar handles it
    };
    let lit_id = graph.strs.get(lit); // None ⇒ literal not interned ⇒ matches nothing
    let is_eq = matches!(op, CompareOp::Eq);
    let mut t = Vec::with_capacity(sc.n);
    let mut valid = Vec::with_capacity(sc.n);
    for &row in ids {
        let i = row as usize;
        let p = present.get(i);
        valid.push(p);
        let eq = p && Some(data[i]) == lit_id;
        t.push(if is_eq { eq } else { !eq });
    }
    Some(VVec::Bool { t, valid })
}

/// Extract a row-independent string scalar from `e` — a string literal or a
/// `$param` bound to one — else `None`. A param bound to anything else (number,
/// null, list, temporal) must NOT reach the interned-id path: that path answers
/// only string-vs-string, and cross-type comparison has its own semantics.
fn scalar_str<'a>(e: &'a CExpr, ctx: &'a Ctx) -> Option<&'a str> {
    match e {
        CExpr::Lit(Lit::Str(s)) => Some(s),
        CExpr::Param(s) => match ctx.params.get(*s) {
            Some(Val::Str(v)) => Some(v),
            _ => None,
        },
        _ => None,
    }
}

/// Extract a row-independent temporal scalar from `e` — a temporal literal
/// (`DATE '…'`) or a `$param` bound to a temporal — else `None`.
fn scalar_temporal(e: &CExpr, ctx: &Ctx) -> Option<crate::temporal::Temporal> {
    match e {
        CExpr::Lit(Lit::Temporal(t)) => Some(*t),
        CExpr::Param(s) => match ctx.params.get(*s) {
            Some(Val::Temporal(t)) => Some(*t),
            _ => None,
        },
        _ => None,
    }
}

/// Vectorized `<temporal col> <op> <temporal scalar>` (either operand order):
/// compare the packed temporal column against a single scalar via the SAME
/// [`compare_vals`] the scalar path uses — so it's byte-identical, including
/// `Eq`/`Ne`, three-valued null handling, and duration/cross-kind → UNKNOWN — but
/// without the per-row `Binding` rebuild + `Env` + expr-tree `eval`. Returns `None`
/// (fall back to scalar) unless exactly one side is a temporal Prop column and the
/// other a temporal scalar (literal or `$param`).
fn temporal_cmp_vec(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    op: CompareOp,
    left: &CExpr,
    right: &CExpr,
) -> Option<VVec> {
    let (prop, scalar, prop_left) = match (left, right) {
        (p @ CExpr::Prop { .. }, other) => (p, scalar_temporal(other, ctx)?, true),
        (other, p @ CExpr::Prop { .. }) => (p, scalar_temporal(other, ctx)?, false),
        _ => return None,
    };
    let CExpr::Prop { var_slot, key_ref } = prop else {
        return None;
    };
    let (elem, ids) = sc.slot(*var_slot)?;
    let (store, kid) = match elem {
        Elem::Node => (&graph.props, ctx.prop_keys[*key_ref].0),
        Elem::Edge => (&graph.edge_props, ctx.prop_keys[*key_ref].1),
    };
    let Some(Column::Temporal { data, present }) = kid.and_then(|k| store.cols.get(k as usize))
    else {
        return None; // not a typed temporal column (Mixed/absent) — scalar handles it
    };
    let scalar_val = Val::Temporal(scalar);
    let mut t = Vec::with_capacity(sc.n);
    let mut valid = Vec::with_capacity(sc.n);
    for &row in ids {
        let i = row as usize;
        // An absent value is a NULL operand → the whole comparison is UNKNOWN for
        // EVERY op, including `=`/`<>` (three-valued logic). The scalar path guards
        // this before `compare_vals` (`is_nullish` → Val::Null), so we must too —
        // otherwise `col <> $p` would wrongly count absent rows.
        if !present.get(i) {
            t.push(false);
            valid.push(false);
            continue;
        }
        let stored = Val::Temporal(data.get(i));
        let (lv, rv) = if prop_left {
            (&stored, &scalar_val)
        } else {
            (&scalar_val, &stored)
        };
        // Both operands present: `compare_vals` yields Bool (Eq/Ne, or an ordered
        // pair) or Null (UNKNOWN — an unordered/cross-kind `< > <= >=`). Map Null to
        // an invalid slot, exactly what `VVec::Gen`+`into_truth` would produce.
        match compare_vals(ctx, op, lv, rv) {
            Val::Bool(b) => {
                t.push(b);
                valid.push(true);
            }
            _ => {
                t.push(false);
                valid.push(false);
            }
        }
    }
    Some(VVec::Bool { t, valid })
}

/// `min`/`max` over a typed temporal column, folded via the canonical total order
/// (`Temporal::cmp_total`, first-seen on ties — identical to the scalar
/// `fold_extreme`) on reconstructed Copy temporals: no `Val` build per row, no
/// scalar accumulator. `None` unless `spec.arg` is a `Prop` over a
/// `Column::Temporal` and `spec.func` is `Min`/`Max`.
/// `sum` over a slice of already-gathered temporal `Val`s: fold DURATIONs
/// component-wise (overflow → loud), fault on any non-DURATION (dates/times aren't
/// summable). Used by the tree-walking [`eval_aggregate`].
fn temporal_values_sum(values: &[Val], ctx: &Ctx) -> Val {
    use crate::temporal::Temporal as T;
    let mut acc: Option<crate::temporal::Duration> = None;
    for v in values {
        let Val::Temporal(T::Duration(d)) = v else {
            ctx.set_fault(FAULT_TEMPORAL_AGG);
            return Val::Null;
        };
        acc = Some(match acc {
            None => *d,
            Some(a) => match a.add(d) {
                Some(s) => s,
                None => {
                    ctx.set_fault(FAULT_DURATION_OVERFLOW);
                    return Val::Null;
                }
            },
        });
    }
    acc.map_or(Val::Null, |d| Val::Temporal(T::Duration(d)))
}

fn temporal_agg(graph: &Graph, ctx: &Ctx, sc: &ScanCols, spec: &CAgg) -> Option<Val> {
    use crate::temporal::{Temporal as T, TemporalKind};
    if !matches!(spec.func, AggFn::Min | AggFn::Max | AggFn::Sum | AggFn::Avg) {
        return None;
    }
    let CExpr::Prop { var_slot, key_ref } = spec.arg.as_ref()? else {
        return None;
    };
    let (elem, ids) = sc.slot(*var_slot)?;
    let (store, kid) = match elem {
        Elem::Node => (&graph.props, ctx.prop_keys[*key_ref].0),
        Elem::Edge => (&graph.edge_props, ctx.prop_keys[*key_ref].1),
    };
    let Some(Column::Temporal { data, present }) = kid.and_then(|k| store.cols.get(k as usize))
    else {
        return None;
    };
    // `avg` over any temporal, and `sum` over a non-DURATION temporal, are loud
    // data exceptions — not a silent null. `avg` needs duration÷count (often
    // non-representable, e.g. avg(P1M,P2M)=P1.5M); dates/times aren't summable.
    if spec.func == AggFn::Avg || (spec.func == AggFn::Sum && data.kind() != TemporalKind::Duration)
    {
        ctx.set_fault(FAULT_TEMPORAL_AGG);
        return Some(Val::Null);
    }
    if let AggFn::Min | AggFn::Max = spec.func {
        let want = if spec.func == AggFn::Min {
            Ordering::Less
        } else {
            Ordering::Greater
        };
        let mut ext: Option<T> = None;
        for &row in ids {
            let i = row as usize;
            if present.get(i) {
                let v = data.get(i);
                ext = Some(match ext {
                    Some(e) if v.cmp_total(&e) == want => v,
                    Some(e) => e,
                    None => v,
                });
            }
        }
        return Some(ext.map_or(Val::Null, Val::Temporal));
    }
    // `sum` over a DURATION column: component-wise fold via the same `Duration::add`
    // as `dur + dur`, so overflow is loud (byte-identical to scalar arithmetic).
    let mut acc: Option<crate::temporal::Duration> = None;
    for &row in ids {
        let i = row as usize;
        if present.get(i) {
            let T::Duration(d) = data.get(i) else {
                unreachable!("kind checked above")
            };
            acc = Some(match acc {
                None => d,
                Some(a) => match a.add(&d) {
                    Some(s) => s,
                    None => {
                        ctx.set_fault(FAULT_DURATION_OVERFLOW);
                        return Some(Val::Null);
                    }
                },
            });
        }
    }
    Some(acc.map_or(Val::Null, |d| Val::Temporal(T::Duration(d))))
}

fn eval_vec(graph: &Graph, ctx: &Ctx, sc: &ScanCols, e: &CExpr) -> VVec {
    let n = sc.n;
    let gen = |e: &CExpr| VVec::Gen(scalar_col(graph, ctx, sc, e));
    match e {
        CExpr::Lit(Lit::Num(x)) => VVec::Num {
            d: vec![*x; n],
            valid: vec![true; n],
        },
        CExpr::Lit(Lit::Bool(b)) => VVec::Bool {
            t: vec![*b; n],
            valid: vec![true; n],
        },
        CExpr::Lit(Lit::Null) => VVec::Num {
            d: vec![f64::NAN; n],
            valid: vec![false; n],
        },
        // A bare variable: a carried value column is taken directly (no per-row
        // binding rebuild); an element column becomes a column of element handles.
        CExpr::Var(slot) => {
            if let Some(v) = sc.val_slot(*slot) {
                VVec::Gen(v.to_vec())
            } else if let Some((elem, ids)) = sc.slot(*slot) {
                VVec::Gen(
                    ids.iter()
                        .map(|&i| match elem {
                            Elem::Node => Val::Node(i),
                            Elem::Edge => Val::Edge(i),
                        })
                        .collect(),
                )
            } else {
                VVec::Gen(vec![Val::Null; n])
            }
        }
        CExpr::Prop { var_slot, key_ref } => match sc.slot(*var_slot) {
            Some((Elem::Node, ids)) => {
                let col = ctx.prop_keys[*key_ref]
                    .0
                    .and_then(|k| graph.props.cols.get(k as usize));
                gather_num(col, ids)
                    .or_else(|| gather_str(col, ids, &graph.strs))
                    .or_else(|| gather_temporal(col, ids))
                    .unwrap_or_else(|| gen(e))
            }
            Some((Elem::Edge, ids)) => {
                let col = ctx.prop_keys[*key_ref]
                    .1
                    .and_then(|k| graph.edge_props.cols.get(k as usize));
                gather_num(col, ids)
                    .or_else(|| gather_str(col, ids, &graph.strs))
                    .or_else(|| gather_temporal(col, ids))
                    .unwrap_or_else(|| gen(e))
            }
            None => gen(e),
        },
        CExpr::Neg(x) => {
            let v = eval_vec(graph, ctx, sc, x);
            // A non-numeric operand → scalar fallback, which raises the type error.
            if matches!(v, VVec::Gen(_)) {
                gen(e)
            } else {
                let (mut d, valid) = v.into_num();
                for v in &mut d {
                    *v = -*v;
                }
                VVec::Num { d, valid }
            }
        }
        CExpr::Arith { head, tail } => {
            // n-ary left-associative fold, column at a time. A non-numeric operand
            // (general column, incl. temporal, OR a boolean) → scalar fallback for
            // the whole node, which raises the ISO type error / does temporal
            // arithmetic per-row rather than coercing to NaN. A Bool VVec must fall
            // back too: `into_num` would coerce it to 0/1 (valid), but the scalar
            // `arith_num` faults on a boolean — and the TS engine throws — so
            // `true + 1` over rows must fault, not compute 2.
            let mut acc = eval_vec(graph, ctx, sc, head);
            if matches!(acc, VVec::Gen(_) | VVec::Bool { .. }) {
                return gen(e);
            }
            for (op, rhs) in tail {
                let r = eval_vec(graph, ctx, sc, rhs);
                if matches!(r, VVec::Gen(_) | VVec::Bool { .. }) {
                    return gen(e);
                }
                acc = arith_vec_step(ctx, *op, acc, r, n);
            }
            acc
        }
        CExpr::Compare { op, left, right } => {
            // Interned-id string equality (`col = / <> literal`) — no per-row bytes.
            if let Some(v) = str_eq_vec(graph, ctx, sc, *op, left, right) {
                return v;
            }
            // Typed `<temporal col> <op> <temporal scalar>` — packed compare, no
            // per-row eval dispatch (byte-identical via `compare_vals`).
            if let Some(v) = temporal_cmp_vec(graph, ctx, sc, *op, left, right) {
                return v;
            }
            let l = eval_vec(graph, ctx, sc, left);
            let r = eval_vec(graph, ctx, sc, right);
            // Numeric fast path when both sides are the same category (Num/Num or
            // Bool/Bool); otherwise the comparison is over strings/identity or is
            // cross-type → scalar fallback.
            match (&l, &r) {
                (VVec::Gen(_), _) | (_, VVec::Gen(_)) => gen(e),
                // A boolean compared with a number is cross-type: the numeric path
                // below would wrongly coerce true/false to 1/0 (so `1 = true` passed
                // a WHERE). Route to the scalar evaluator, which gives the LPG
                // cross-type result (eq → false, order → null) — matching the
                // const-folded and property-vs-bool paths and the TS engine.
                (VVec::Num { .. }, VVec::Bool { .. }) | (VVec::Bool { .. }, VVec::Num { .. }) => {
                    gen(e)
                }
                _ => {
                    let (ld, lv) = l.into_num();
                    let (rd, rv) = r.into_num();
                    let mut t = Vec::with_capacity(n);
                    let mut valid = Vec::with_capacity(n);
                    for i in 0..n {
                        valid.push(lv[i] && rv[i]);
                        let a = ld[i];
                        let b = rd[i];
                        t.push(match op {
                            CompareOp::Eq => a == b,
                            CompareOp::Ne => a != b,
                            CompareOp::Lt => a < b,
                            CompareOp::Gt => a > b,
                            CompareOp::Le => a <= b,
                            CompareOp::Ge => a >= b,
                        });
                    }
                    VVec::Bool { t, valid }
                }
            }
        }
        // `size(x)` over a carried VALUE column — the shape every group variable
        // is read through (`size(e)`, and the predicate `size(e) >= 2`). Without
        // this it falls to `scalar_col`, which rebuilds a whole `Binding` per row
        // just to count a list, which was most of why the columnar path lost to
        // the scalar matcher on quantified units.
        CExpr::Scalar { func, args }
            if *func == ScalarFn::Size
                && matches!(&args[0], CExpr::Var(s) if sc.val_slot(*s).is_some()) =>
        {
            let CExpr::Var(slot) = &args[0] else {
                unreachable!("guarded above")
            };
            let vals = sc.val_slot(*slot).expect("guarded above");
            let mut d = Vec::with_capacity(n);
            let mut valid = Vec::with_capacity(n);

            for v in vals {
                match v {
                    Val::List(items) => {
                        d.push(items.len() as f64);
                        valid.push(true);
                    }
                    Val::Str(s) => {
                        d.push(s.encode_utf16().count() as f64);
                        valid.push(true);
                    }
                    Val::Path(p) => {
                        d.push(p.edges.len() as f64);
                        valid.push(true);
                    }
                    _ => {
                        d.push(f64::NAN);
                        valid.push(false);
                    }
                }
            }

            VVec::Num { d, valid }
        }
        CExpr::Scalar { func, args } if args.len() == 1 && unary_math(*func).is_some() => {
            let f = unary_math(*func).unwrap();
            let (mut d, valid) = eval_vec(graph, ctx, sc, &args[0]).into_num();
            for v in &mut d {
                *v = f(*v);
            }
            VVec::Num { d, valid }
        }
        CExpr::Not(x) => {
            let tr = eval_vec(graph, ctx, sc, x).into_truth();
            kleene_vec(tr.iter().map(|&t| not3(t)))
        }
        CExpr::And(items) => {
            let mut acc = eval_vec(graph, ctx, sc, &items[0]).into_truth();
            for e in &items[1..] {
                let b = eval_vec(graph, ctx, sc, e).into_truth();
                for i in 0..n {
                    acc[i] = and3(acc[i], b[i]);
                }
            }
            kleene_vec(acc.into_iter())
        }
        CExpr::Or(items) => {
            let mut acc = eval_vec(graph, ctx, sc, &items[0]).into_truth();
            for e in &items[1..] {
                let b = eval_vec(graph, ctx, sc, e).into_truth();
                for i in 0..n {
                    acc[i] = or3(acc[i], b[i]);
                }
            }
            kleene_vec(acc.into_iter())
        }
        CExpr::Xor(items) => {
            let mut acc = eval_vec(graph, ctx, sc, &items[0]).into_truth();
            for e in &items[1..] {
                let b = eval_vec(graph, ctx, sc, e).into_truth();
                for i in 0..n {
                    acc[i] = xor3(acc[i], b[i]);
                }
            }
            kleene_vec(acc.into_iter())
        }
        CExpr::IsNull { expr, negated } => {
            let (_, valid) = eval_vec(graph, ctx, sc, expr).into_num();
            let t = valid
                .iter()
                .map(|&v| if *negated { v } else { !v })
                .collect();
            VVec::Bool {
                t,
                valid: vec![true; n],
            }
        }
        _ => gen(e),
    }
}

/// Build a `VVec::Bool` from a Kleene-truth stream (`None` → invalid/UNKNOWN).
fn kleene_vec(it: impl Iterator<Item = Truth>) -> VVec {
    let mut t = Vec::new();
    let mut valid = Vec::new();
    for tr in it {
        match tr {
            Some(b) => {
                t.push(b);
                valid.push(true);
            }
            None => {
                t.push(false);
                valid.push(false);
            }
        }
    }
    VVec::Bool { t, valid }
}

/// Materialize the matched rows of a fixed-length path into columns. An isolated
/// node is a tight label-bucket scan; a traversal is a batched adjacency
/// expansion — walk each frontier node's edges and push straight into the
/// columns, multiplying rows by matching neighbors, with no matcher recursion or
/// per-edge bind/unset. Returns `None` (→ scalar path) for var-length
/// quantifiers or a slot a path binds twice (a self-join). `cap` stops early.
fn lit_to_idxkey(lit: &Lit) -> Option<crate::graph::IdxKey> {
    use crate::graph::IdxKey;
    match lit {
        Lit::Str(s) => Some(IdxKey::Str(s.as_str().into())),
        Lit::Num(n) => Some(IdxKey::Num(*n)),
        Lit::Bool(b) => Some(IdxKey::Bool(*b)),
        // Temporals key via the same monotonic encoding as the stored column, so a
        // `WHERE r.vf <= DATE '…'` seeks (Duration → None, still a scan).
        Lit::Temporal(t) => t.index_key().map(|(k, key)| IdxKey::Temporal(k, key)),
        Lit::Null => None,
    }
}

/// A runtime value as an index key (nulls/lists/elements aren't indexable).
/// A runtime value as an index key — see [`Value::index_key`], which both
/// engines now share.
fn val_to_idxkey(v: &Val) -> Option<crate::graph::IdxKey> {
    v.index_key()
}

/// The index key an expression contributes to a seek: an inline literal, or a
/// `$param` resolved against the current bindings at execute time. Resolving
/// params here is what lets `WHERE v.k = $x` (not just `= 'lit'`) hit the index —
/// matching the TS engine, whose planner seeks on params too.
fn expr_to_idxkey(e: &CExpr, ctx: &Ctx) -> Option<crate::graph::IdxKey> {
    match e {
        CExpr::Lit(lit) => lit_to_idxkey(lit),
        CExpr::Param(slot) => val_to_idxkey(ctx.params.get(*slot)?),
        _ => None,
    }
}

/// The (var slot, index path) a comparison's left side addresses: a bare
/// `var.key` → (slot, `"key"`), a nested `var.a.b…` (a `Field` chain rooted at a
/// `Prop`) → (slot, `"a.b…"`) — the dotted path the [dotted-path index] is keyed
/// by, so `WHERE n.meta.city = $x` can seek a `meta.city` index instead of
/// scanning. Any other shape (a computed base, a non-`Prop` root) → `None`.
fn prop_path(left: &CExpr, graph: &Graph, ctx: &Ctx, edge: bool) -> Option<(usize, String)> {
    match left {
        CExpr::Prop { var_slot, key_ref } => Some((
            *var_slot,
            prop_name(graph, ctx, *key_ref, edge)?.to_string(),
        )),
        // A collapsed stored-field read (`n.meta.city`) — the common nested form.
        CExpr::PropField {
            var_slot,
            root_key_ref,
            descent,
        } => {
            let mut path = prop_name(graph, ctx, *root_key_ref, edge)?.to_string();
            for seg in descent {
                path.push('.');
                path.push_str(seg);
            }
            Some((*var_slot, path))
        }
        // A `Field` that didn't collapse (a computed base) — still walk it.
        CExpr::Field { base, name, .. } => {
            let (slot, mut path) = prop_path(base, graph, ctx, edge)?;
            path.push('.');
            path.push_str(name);
            Some((slot, path))
        }
        _ => None,
    }
}

/// A `var.key OP <literal-or-$param>` comparison, as (var slot, key ref, op,
/// resolved index key). The constant is resolved via [`expr_to_idxkey`] so
/// params seek as well as literals.
///
/// Either ORDER, with the operator flipped to match: `5 <= u.n` is `u.n >= 5`.
/// This used to read only the left operand, so a conjunction written
/// `5 <= u.n AND 9 >= u.n` lost its range seek and scanned — 197x on a 20k-vertex
/// graph — while the identical `u.n >= 5 AND u.n <= 9` seeked. The single-
/// comparison path in `prop_index_hint` had the same gap; this is the grouped
/// path, which is separate code and needed the same fix.
fn cmp_bound(e: &CExpr, ctx: &Ctx) -> Option<(usize, usize, CompareOp, crate::graph::IdxKey)> {
    if let CExpr::Compare { op, left, right } = e {
        if let CExpr::Prop { var_slot, key_ref } = left.as_ref() {
            let key = expr_to_idxkey(right, ctx)?;

            return Some((*var_slot, *key_ref, *op, key));
        }
        if let CExpr::Prop { var_slot, key_ref } = right.as_ref() {
            let key = expr_to_idxkey(left, ctx)?;

            return Some((*var_slot, *key_ref, flip_compare(*op), key));
        }
    }
    None
}

// --- index-seeded scanning, expansion, and vectorized aggregation ---
mod scan;
mod seek_lower;
#[cfg(test)]
mod seek_lower_tests;
use scan::*;

// Query-shape fast-paths (count / grouped / parallel shortcuts).
mod fastpath;
use fastpath::*;

/// Vectorized executor for a whole linear pipeline: `MATCH <path> (WITH …)+
/// RETURN …`. Threads a single columnar frame stage-to-stage — carrying element
/// columns forward so prop reads/filters/ORDER BY past a `WITH` stay vectorized,
/// instead of round-tripping each stage through `Vec<Binding>`. Returns `None`
/// (→ scalar `run_linear`) unless: one leading non-optional single-path MATCH, at
/// least one intermediate (non-aggregating) WITH, a terminal RETURN, and nothing
/// else (no extra MATCH / mutation / subquery clause, no var-length, no self-join).
fn vectorized_linear(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<RowSet> {
    // The `count(*)` shortcuts (`try_count_star` / `try_count_edges`) are applied
    // earlier, in `run_part`, ahead of the parallel path — so they aren't repeated
    // here.
    // Validate the whole clause shape *before* any scan work, so a non-pipeline
    // query bails for free (no wasted build_scan) and keeps the entry path.
    let (first, rest) = linear.clauses.split_first()?;
    let CClause::Match {
        optional: false,
        patterns,
        where_,
        scope_len,
        ..
    } = first
    else {
        return None;
    };
    if patterns.len() != 1 {
        return None;
    }
    let (last, mid) = rest.split_last()?;
    let CClause::Return(last_proj) = last else {
        return None;
    };
    // Middle clauses are WITHs or expanding MATCHes.
    let mid_ok = mid
        .iter()
        .all(|c| matches!(c, CClause::With { .. } | CClause::Match { .. }));
    if !mid_ok {
        return None;
    }
    // A plain `MATCH … RETURN` (no intermediate WITH) normally stays on the scalar
    // entry path so a `RETURN … LIMIT n` keeps its row-by-row early-out. But an
    // aggregate (`count`/`sum`/`avg`/group-by) scans the whole match regardless —
    // there's no early-out to lose — so route it through the vectorized frame for
    // the de-boxed columnar win on filtered counts/sums. (The Arrow fast path
    // already covers non-aggregate plain `MATCH … RETURN`, but bars aggregates;
    // this fills exactly that gap.)
    if mid.is_empty() && !last_proj.aggregating {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    let filter = |sc: &mut ScanCols, w: &CExpr| {
        let keep: Vec<bool> = eval_vec(graph, &ctx, sc, w)
            .into_truth()
            .iter()
            .map(|t| *t == Some(true))
            .collect();
        compact(sc, &keep);
    };
    let mut sc = build_scan(
        graph,
        &ctx,
        &patterns[0],
        *scope_len,
        None,
        where_.as_ref(),
        None,
    )?;
    if let Some(w) = where_ {
        filter(&mut sc, w);
    }
    for c in mid {
        match c {
            CClause::With {
                projection, where_, ..
            } => {
                sc = with_frame(graph, &ctx, &sc, projection)?;
                if let Some(w) = where_ {
                    filter(&mut sc, w);
                }
            }
            CClause::Match {
                patterns,
                where_,
                scope_len,
                optional,
                ..
            } => {
                if patterns.len() != 1 {
                    return None;
                }
                if *optional {
                    // A clause-level WHERE on an OPTIONAL clause would have to
                    // null-fill (not drop) rows that fail it — not handled; scalar.
                    if where_.is_some() {
                        return None;
                    }
                    sc = expand_frame_optional(graph, &ctx, &sc, &patterns[0], *scope_len)?;
                } else {
                    sc = expand_frame(graph, &ctx, &sc, &patterns[0], *scope_len)?;
                    if let Some(w) = where_ {
                        filter(&mut sc, w);
                    }
                }
            }
            _ => return None,
        }
    }
    let proj = last_proj;
    let cols = project_frame_cols(graph, &ctx, &sc, proj)?;
    let nrows = cols.first().map_or(0, |c| c.len());
    let mut rs = RowSet::new(proj.out_names.clone());
    for i in 0..nrows {
        rs.push_row(cols.iter().map(|c| val_to_value(graph, &c[i])));
    }
    // A data exception during vectorized eval can't return `Err` from here; fall
    // back to the scalar path (this query shape is read-only, so re-running is
    // safe), which re-evaluates and surfaces the `CodeError`.
    if ctx.faulted() {
        return None;
    }
    Some(rs)
}

/// Project a terminal RETURN directly to output `Value` rows. For the common
/// non-aggregating, non-ordered case this streams straight to one `Vec<Value>`
/// per row — no intermediate `Binding` and no later Val→Value conversion (the
/// dominant materialization cost). Aggregating/ordered fall back to the binding
/// accumulator (few/already-sorted rows) and convert.
fn project_to_rows(
    graph: &Graph,
    ctx: &Ctx,
    incoming: &[Binding],
    matches: &[&CClause],
    proj: &CProjection,
) -> RowSet {
    // A zero LIMIT emits no rows, so the projection never runs. The rule the engine
    // already follows for a non-zero LIMIT is "project exactly the rows you emit" —
    // `RETURN 1/(n.n - 7) AS x LIMIT 1` returns the first row rather than faulting on
    // the second, in BOTH engines. LIMIT 0 emits nothing, so it projects nothing;
    // faulting here would make LIMIT 0 the one limit that evaluates discarded rows.
    // (Not a SQL rule: ISO GQL's `<order by and page statement>` is a statement in
    // its own right that may PRECEDE the RETURN, where paging trims the binding table
    // before any projection — this is what that form yields. We implement only the
    // trailing form. Postgres is not the reference and does not agree: it
    // constant-folds `1/0` at plan time and raises even under LIMIT 0.)
    if proj.limit_val(ctx) == Some(0) {
        return RowSet::new(proj.out_names.clone());
    }
    if use_vec() {
        // Plain projection: transpose straight from the typed `VVec`s to the
        // RowSet (no intermediate `Vec<Val>` columns / second conversion pass).
        if let Some(rs) = vectorized_rowset(graph, ctx, incoming, matches, proj) {
            return rs;
        }
        // Aggregating / ORDER BY / DISTINCT: materialized columns, then transpose.
        if let Some(cols) = vectorized_cols(graph, ctx, incoming, matches, proj) {
            // Terminal output: flatten element handles to their ids.
            let nrows = cols.first().map_or(0, |c| c.len());
            let mut rs = RowSet::new(proj.out_names.clone());
            for i in 0..nrows {
                rs.push_row(cols.iter().map(|c| val_to_value(graph, &c[i])));
            }
            return rs;
        }
        // A vectorized attempt that bailed on the intermediate budget set the fault
        // and returned None. Don't fall through to the scalar driver — it would
        // re-enumerate the same runaway cross-product. The fault is drained at the
        // statement boundary, so the (empty) RowSet is discarded and the
        // E_RESOURCE_EXHAUSTED surfaces.
        if ctx.faulted() {
            return RowSet::new(proj.out_names.clone());
        }
    }
    let mut rs = RowSet::new(proj.out_names.clone());
    if proj.aggregating || !proj.order_by.is_empty() {
        // Few / already-sorted rows: reuse the binding accumulator, then pour
        // each projected binding's cells into the flat buffer.
        for b in project_matches(graph, ctx, incoming, matches, proj) {
            rs.push_row((0..proj.out_len).map(|i| {
                b.get(i)
                    .map(|v| val_to_value(graph, v))
                    .unwrap_or(Value::Null)
            }));
        }
        return rs;
    }
    // Fast path: project each row straight into the flat cell buffer — no
    // intermediate per-row Vec, no second conversion pass.
    let cap = proj.limit_val(ctx).map(|l| proj.skip_val(ctx) + l);
    let mut seen: FxHashSet<String> = FxHashSet::default();
    let simple = single_simple_clause(matches);
    for inb in incoming {
        let mut work = inb.clone();
        // The row-pushing sink (shared by the monomorphized and dyn drivers).
        let mut push = |b: &Binding| -> bool {
            if proj.star {
                rs.push_row(proj.star_cols.iter().map(|&s| {
                    b.get(s)
                        .map(|v| val_to_value(graph, v))
                        .unwrap_or(Value::Null)
                }));
            } else {
                let env = Env::new(graph, ctx, b);
                rs.push_row(
                    proj.items
                        .iter()
                        .map(|item| val_to_value(graph, &eval_item(&env, item))),
                );
            }
            if proj.distinct && !seen.insert(value_row_key(rs.row(rs.nrows - 1))) {
                rs.pop_row();
                return true;
            }
            cap.is_none_or(|c| rs.nrows < c) // stop once enough collected
        };
        let cont = match simple {
            Some((path, cwhere, cwhere_prog, scope_len)) => {
                work.resize(scope_len);
                match_one_path(graph, ctx, path, &mut work, &mut |b| {
                    if !where_keep(&Env::new(graph, ctx, b), cwhere, cwhere_prog) {
                        return true;
                    }
                    push(b)
                })
            }
            None => drive_matches(graph, ctx, matches, 0, &mut work, &mut |b| push(b)),
        };
        if !cont {
            break;
        }
    }
    rs.apply_skip_limit(proj.skip_val(ctx), proj.limit_val(ctx));
    rs
}

/// Materialize the binding stream from `incoming × pending matches` (needed
/// before a write clause, which mutates per row).
fn materialize_matches(
    graph: &Graph,
    ctx: &Ctx,
    incoming: &[Binding],
    matches: &[&CClause],
) -> Vec<Binding> {
    let mut out = Vec::new();
    for inb in incoming {
        let mut work = inb.clone();
        drive_matches(graph, ctx, matches, 0, &mut work, &mut |b| {
            out.push(b.clone());
            true
        });
    }
    out
}

// --- projection, linear query & set-ops, write execution ---
mod statement;
use statement::*;

/// A prepared (lowered) query: compile once, execute many times with different
/// params against any graph. Parameters slot in positionally at execute time.
pub struct Prepared {
    plan: CQuery,
    /// param slot → name (the order positional args are bound in).
    param_names: Vec<String>,
}

impl Prepared {
    pub fn execute(&self, graph: &mut Graph, params: &Params) -> CodeResult<RowSet> {
        run_cquery(&self.plan, graph, &positional(&self.param_names, params)?)
    }
    /// Execute and return the result as an Apache Arrow columnar blob (see
    /// [`crate::arrow`]) — the zero-copy carrier for the FFI / wasm boundary.
    #[cfg(feature = "arrow")]
    pub fn execute_arrow(&self, graph: &mut Graph, params: &Params) -> CodeResult<Vec<u8>> {
        run_cquery_arrow(&self.plan, graph, &positional(&self.param_names, params)?)
    }
}

/// Parse and lower a query into a reusable [`Prepared`] plan. Only the linear
/// query grammar is preparable — a transaction-control command (`START
/// TRANSACTION`/`COMMIT`/`ROLLBACK`) has no reusable plan, so it is parsed via the
/// query grammar here and surfaces the usual "expected a clause" syntax error;
/// run it through the one-shot query path instead.
pub fn prepare(text: &str) -> Result<Prepared, SyntaxError> {
    prepare_with_max_chain(text, super::parser::DEFAULT_MAX_CHAIN)
}

/// Like [`prepare`] but with a caller-supplied operator-chain ceiling (see
/// `parser::DEFAULT_MAX_CHAIN`) — the prepared-statement analogue of
/// `parse_with_max_chain`, honouring the native `maxOperatorChain` option.
/// The lowered plan alone, for tests that inspect plan structure rather than
/// run it — notably the seek-lowering collapse tests.
#[cfg(test)]
pub(crate) fn prepare_plan(text: &str) -> Result<super::plan::CQuery, SyntaxError> {
    Ok(lower(&super::parser::parse_query_with_max_chain(
        text,
        super::parser::DEFAULT_MAX_CHAIN,
    )?)
    .0)
}

pub fn prepare_with_max_chain(text: &str, max_chain: usize) -> Result<Prepared, SyntaxError> {
    let query = super::parser::parse_query_with_max_chain(text, max_chain)?;
    let (plan, param_names) = lower(&query);
    Ok(Prepared { plan, param_names })
}

/// Execute a prepared graph-level INVARIANT query directly against the staged
/// graph, WITHOUT opening a per-statement auto-commit transaction frame. An
/// invariant runs from inside `commit_tx`/`run_deferred_checks`, where the frame
/// [`run_cquery`] normally opens would recurse straight back into the very commit
/// path that invoked it. The invariant is a `MATCH…RETURN` assertion (no writes),
/// so skipping the frame is sound — the caller scans the returned rows for a
/// boolean-`false` cell (`false`-only-fails). Bound with empty params (a whole-
/// graph invariant takes no `$params`; a query that references one surfaces the
/// usual missing-parameter error).
pub fn run_invariant(plan: &Prepared, graph: &mut Graph) -> CodeResult<RowSet> {
    let params = positional(&plan.param_names, &Params::new())?;
    run_cquery_body(&plan.plan, graph, &params)
}

/// Does a query mutate the graph (contain any INSERT/MERGE/SET/REMOVE/DELETE)?
/// Used to reject a write statement inside a READ ONLY transaction. Mirrors the
/// TS `queryHasWrite` over the same clause set.
fn query_has_write(q: &Query) -> bool {
    q.parts.iter().any(|p| {
        p.clauses.iter().any(|c| {
            matches!(
                c,
                Clause::Insert(_)
                    | Clause::Merge(_)
                    | Clause::Set(_)
                    | Clause::Remove(_)
                    | Clause::Delete { .. }
            )
        })
    })
}

/// Execute an ISO GQL transaction-control command by driving the graph's
/// transaction frame. Returns an empty [`RowSet`] (no rows/columns), like a
/// write-only query. ISO semantics are enforced here — NOT in the core primitives:
///  - `START TRANSACTION` while one is active → `E_INVALID_GRAPH_OP` (no nesting);
///    the depth reflects only explicit transactions, since a TxControl is not a
///    write and so is never wrapped in a per-statement auto-commit frame.
///  - `COMMIT`/`ROLLBACK` with no active transaction → `E_INVALID_GRAPH_OP`; the
///    depth is checked here so ROLLBACK is symmetric with COMMIT without changing
///    `Graph::rollback_tx`'s idempotent contract.
///  - the READ ONLY access mode is recorded on the graph (cleared on
///    commit/rollback) for a later write statement to consult.
fn run_tx_control(tx: &TxControl, graph: &mut Graph) -> CodeResult<RowSet> {
    match tx.kind {
        TxKind::Start => {
            if graph.in_transaction() {
                return Err(CodeError::new(
                    ErrorCode::InvalidGraphOp,
                    "START TRANSACTION: a transaction is already active",
                ));
            }
            graph.begin_tx();
            graph.set_tx_read_only(matches!(tx.access_mode, Some(AccessMode::ReadOnly)));
        }
        TxKind::Commit => {
            if !graph.in_transaction() {
                return Err(CodeError::new(
                    ErrorCode::InvalidGraphOp,
                    "COMMIT: no active transaction",
                ));
            }
            let result = graph.commit_tx();
            graph.set_tx_read_only(false);
            if let Err(e) = result {
                return Err(tx_commit_error(e));
            }
        }
        TxKind::Rollback => {
            if !graph.in_transaction() {
                return Err(CodeError::new(
                    ErrorCode::InvalidGraphOp,
                    "ROLLBACK: no active transaction",
                ));
            }
            graph.rollback_tx();
            graph.set_tx_read_only(false);
        }
    }
    Ok(RowSet::new(Vec::new()))
}

/// Reject a write statement running inside a READ ONLY transaction, before it
/// applies (statement-level check — no mutator is touched). A read query is
/// always allowed.
fn enforce_read_only(graph: &Graph, q: &Query) -> CodeResult<()> {
    if graph.tx_read_only() && query_has_write(q) {
        return Err(CodeError::new(
            ErrorCode::InvalidGraphOp,
            "write statement rejected: the active transaction is READ ONLY",
        ));
    }
    Ok(())
}

impl Statement {
    /// Lower and execute in one call (no plan reuse). A linear query runs as usual;
    /// a transaction-control command drives the session's transaction frame and
    /// returns no rows. This is the entry the FFI query path calls after
    /// [`super::parse`].
    pub fn execute(&self, graph: &mut Graph, params: &Params) -> CodeResult<RowSet> {
        match self {
            Self::Query(q) => {
                enforce_read_only(graph, q)?;
                q.execute(graph, params)
            }
            Self::Tx(tx) => run_tx_control(tx, graph),
        }
    }

    /// Lower and execute, returning an Apache Arrow columnar blob. A
    /// transaction-control command produces an empty column blob.
    #[cfg(feature = "arrow")]
    pub fn execute_arrow(&self, graph: &mut Graph, params: &Params) -> CodeResult<Vec<u8>> {
        match self {
            Self::Query(q) => {
                enforce_read_only(graph, q)?;
                q.execute_arrow(graph, params)
            }
            Self::Tx(tx) => {
                let rs = run_tx_control(tx, graph)?;
                Ok(crate::arrow::to_arrow(&rs))
            }
        }
    }
}

impl super::ast::Query {
    /// Lower and execute in one call (no plan reuse). Keeps the simple
    /// `parse(q)?.execute(graph, &params)` path; reuse a [`Prepared`] for speed.
    pub fn execute(&self, graph: &mut Graph, params: &Params) -> CodeResult<RowSet> {
        let (plan, param_names) = lower(self);
        run_cquery(&plan, graph, &positional(&param_names, params)?)
    }
    /// Lower and execute, returning an Apache Arrow columnar blob.
    #[cfg(feature = "arrow")]
    pub fn execute_arrow(&self, graph: &mut Graph, params: &Params) -> CodeResult<Vec<u8>> {
        let (plan, param_names) = lower(self);
        run_cquery_arrow(&plan, graph, &positional(&param_names, params)?)
    }
}

#[cfg(test)]
mod path_value_tests {
    use super::*;
    use crate::ndjson;
    use crate::query::RowSet;

    // a —KNOWS(e1)→ b —KNOWS(e2)→ c. Dense ids follow NDJSON insertion order, so
    // vertices are 0,1,2 and edges 0,1.
    const NDJSON: &str = concat!(
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"name":"A"}}"#,
        "\n",
        r#"{"type":"node","id":"b","labels":["P"],"properties":{"name":"B"}}"#,
        "\n",
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"name":"C"}}"#,
        "\n",
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["KNOWS"],"properties":{}}"#,
        "\n",
        r#"{"type":"edge","id":"e2","from":"b","to":"c","labels":["KNOWS"],"properties":{}}"#,
    );

    /// A `Val::Path` serializes to `{vertices, edges, length}` — byte-for-byte the
    /// TS `Path.toJSON()` shape (see packages/core/src/core/Path.test.ts): vertices
    /// and edges as the rich element maps (sorted labels/props, fixed field order),
    /// `length` the hop count.
    #[test]
    fn path_serializes_to_vertices_edges_length() {
        let g = ndjson::decode(NDJSON).unwrap();
        let path = Val::path(vec![0, 1, 2], vec![0, 1]);

        let mut rs = RowSet::new(vec!["p".to_string()]);
        rs.push_row([val_to_value(&g, &path)]);

        assert_eq!(
            rs.to_json(),
            concat!(
                r#"{"columns":["p"],"rows":[[{"vertices":["#,
                r#"{"id":"a","labels":["P"],"properties":{"name":"A"}},"#,
                r#"{"id":"b","labels":["P"],"properties":{"name":"B"}},"#,
                r#"{"id":"c","labels":["P"],"properties":{"name":"C"}}],"#,
                r#""edges":["#,
                r#"{"id":"e1","from":"a","to":"b","labels":["KNOWS"],"properties":{}},"#,
                r#"{"id":"e2","from":"b","to":"c","labels":["KNOWS"],"properties":{}}],"#,
                r#""length":2}]]}"#,
            )
        );
    }

    /// The DISTINCT/grouping key is structural: same vertices + edges → same key.
    #[test]
    fn path_val_key_is_structural() {
        let same_a = Val::path(vec![0, 1, 2], vec![0, 1]);
        let same_b = Val::path(vec![0, 1, 2], vec![0, 1]);
        let diff = Val::path(vec![0, 1], vec![0]);

        let key = |v: &Val| {
            let mut s = String::new();
            val_key(v, &mut s);
            s
        };

        assert_eq!(key(&same_a), key(&same_b));
        assert_ne!(key(&same_a), key(&diff));
    }
}
