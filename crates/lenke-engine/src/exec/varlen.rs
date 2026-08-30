use super::nested::*;
use super::render::*;
use super::*;
use crate::batch::{Batch, Col};
use crate::ir::{Expr, Plan};
use crate::store::Store;
use crate::value::Value;

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
pub(super) fn var_length(
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
    // the host. Cap the total emitted rows at `limits.trail` (the TS engine's guard, same
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
pub(super) fn rep_pred_ok(
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
pub(super) fn until_ok(pred: &Expr, store: &Store, endpoint_slot: usize, v: u32) -> bool {
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
pub(super) fn eval_node_bool(pred: &Expr, store: &Store, v: u32) -> Option<bool> {
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
pub(super) trait VarlenEmit {
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
/// (matching the TS engine, which completes them).
pub(super) struct DistinctEndpointEmit {
    pub(super) seen: Vec<bool>,
    pub(super) out: Vec<u32>,
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
pub(super) fn run_varlen<S: VarlenEmit>(
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
pub(super) fn try_stream_varlen_json(
    plan: &Plan,
    store: &Store,
    gql: bool,
) -> Option<Result<String, String>> {
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
pub(super) enum Enter {
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
pub(super) fn varlen_enter<S: VarlenEmit>(
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
pub(super) fn adj_nth(
    store: &Store,
    v: u32,
    dir: Dir,
    i: usize,
) -> Option<(bool, crate::store::Adj)> {
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
pub(super) fn varlen_walk<S: VarlenEmit>(
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
pub(super) fn varlen_dfs<S: VarlenEmit>(
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
pub(super) fn shortest_path(
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
pub(super) fn edge_pred_ok(pred: Option<&Expr>, store: &Store, eid: u32) -> bool {
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
/// the `k` smallest distinct lengths (`group`). Mirrors the TS engine's `shortest_k_walk`;
/// the endpoint's own label/property filter is a `Filter` above this, so it selects
/// k per endpoint here and the filter narrows afterward.
#[allow(clippy::too_many_arguments)]
pub(super) fn shortest_k_path(
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
pub(super) fn collect_trails(
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
pub(super) fn first_pred_chain(
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
/// same cost the TS engine's `enumerate_shortest_paths` pays; no case in scope hits it.
pub(super) fn enumerate_shortest_paths(
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
pub(super) fn push_path(
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
