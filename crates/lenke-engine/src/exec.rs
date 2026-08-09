//! Execution: pull a batch up through the plan, then materialize the projection.
//!
//! Expression evaluation is columnar — `eval` produces a `Col` over the whole
//! batch, reading typed storage columns in bulk where it can. It calls the value
//! contract for every comparison and equality; it never restates those rules.
//! This is the lineage-FREE strategy; the lineage-preserving strategy for the
//! same operators lands with the operators (path/tags) that need it.

use std::collections::{HashMap, VecDeque};

use crate::batch::{Batch, Col, Lineage};
use crate::ir::{Agg, AggFn, CompareOp, Dir, Expr, Plan};
use crate::store::{Column, Store};
use crate::value::{self, Value};

/// A materialized result: column names and rows of values. `Value` intentionally
/// has no `PartialEq` (f64/NaN policy lives in the value contract, not a derive),
/// so compare results through `value::equals`/`cmp_total`, not `==`.
#[derive(Debug)]
pub struct Rows {
    pub names: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

/// Run `plan` over `store`, returning materialized rows. Output column names come
/// from the outermost naming operator (`Project` or `Aggregate`, seen through
/// `Distinct`/`OrderPage`); a plan with none surfaces slot 0 under a single
/// implicit column so partial plans stay runnable in tests.
#[must_use]
pub fn run(plan: &Plan, store: &Store) -> Rows {
    // Lineage is plan-global: if anything reads the path, the whole plan tracks
    // it (Scan seeds, Expand extends); otherwise no operator builds a sidecar and
    // the query pays nothing for lineage.
    let track = needs_lineage(plan);
    let batch = pull(plan, store, track);
    let n = batch.rows();
    match output_names(plan) {
        Some(names) => {
            let rows = (0..n)
                .map(|i| batch.slots.iter().map(|c| c.value_at(i)).collect())
                .collect();
            Rows { names, rows }
        }
        None => {
            let slot0 = batch.slot(0);
            let rows = (0..n).map(|i| vec![slot0.value_at(i)]).collect();
            Rows {
                names: vec!["_".to_string()],
                rows,
            }
        }
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
        Plan::Distinct { input } | Plan::OrderPage { input, .. } => output_names(input),
        _ => None,
    }
}

/// Whether any expression in the plan reads the path (`Expr::Path`) — the signal
/// that lineage must be tracked. Computed once, for the whole plan.
fn needs_lineage(plan: &Plan) -> bool {
    fn reads_path(e: &Expr) -> bool {
        match e {
            Expr::Path => true,
            Expr::Compare { left, right, .. } => reads_path(left) || reads_path(right),
            Expr::Not(x) => reads_path(x),
            Expr::And(a, b) | Expr::Or(a, b) => reads_path(a) || reads_path(b),
            Expr::Slot(_) | Expr::Prop { .. } | Expr::Lit(_) => false,
        }
    }
    match plan {
        Plan::Scan { .. } => false,
        Plan::Expand { input, .. }
        | Plan::VarLength { input, .. }
        | Plan::ShortestPath { input, .. }
        | Plan::Distinct { input } => needs_lineage(input),
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
        Plan::Join { left, right, .. } => needs_lineage(left) || needs_lineage(right),
    }
}

/// Pull a batch up through a (non-terminal) plan node. `track` is the plan-global
/// lineage decision: when true, row-producing operators build the path sidecar.
fn pull(plan: &Plan, store: &Store, track: bool) -> Batch {
    match plan {
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
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
        } => expand(
            &pull(input, store, track),
            store,
            *from,
            *dir,
            edge_label.as_deref(),
        ),
        Plan::Filter { input, pred } => {
            let batch = pull(input, store, track);
            let mask = eval(pred, store, &batch);
            let keep: Vec<usize> = match &mask {
                Col::Bool(bs) => (0..bs.len()).filter(|&i| bs[i]).collect(),
                other => (0..other.len())
                    .filter(|&i| other.value_at(i).is_true())
                    .collect(),
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
            trail,
        } => var_length(
            &pull(input, store, track),
            store,
            *from,
            *dir,
            edge_label.as_deref(),
            *min,
            *max,
            *trail,
        ),
        Plan::ShortestPath {
            input,
            from,
            dir,
            edge_label,
            max,
        } => shortest_path(
            &pull(input, store, track),
            store,
            *from,
            *dir,
            edge_label.as_deref(),
            *max,
        ),
        Plan::Aggregate { input, keys, aggs } => {
            aggregate(&pull(input, store, track), store, keys, aggs)
        }
        Plan::OrderPage {
            input,
            keys,
            skip,
            limit,
        } => order_page(&pull(input, store, track), store, keys, *skip, *limit),
        Plan::Project { input, items } => {
            // Project produces a batch whose slots ARE the projected columns, so
            // an operator above it (Distinct, OrderPage) works on the output
            // values, not the pre-projection bindings.
            let batch = pull(input, store, track);
            let cols = items.iter().map(|(_, e)| eval(e, store, &batch)).collect();
            Batch::of(cols)
        }
        Plan::Distinct { input } => {
            let batch = pull(input, store, track);
            let n = batch.rows();
            let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
            let mut buf = Vec::new();
            let keep: Vec<usize> = (0..n)
                .filter(|&i| {
                    // Key the whole row across every slot, via the grouping
                    // notion (NaN/-0 collapse) — never predicate equality. The
                    // buffer is reused across rows; only a NEW row allocates.
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
        Plan::Join { left, right, on } => {
            hash_join(&pull(left, store, track), &pull(right, store, track), on)
        }
    }
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
    let mut index: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
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

/// Sort the batch by `keys` (stable; ascending via `cmp_total`, descending its
/// reverse), then keep the window `[skip, skip+limit)`. Reorders every slot
/// together, so bound variables stay row-aligned.
fn order_page(
    batch: &Batch,
    store: &Store,
    keys: &[crate::ir::SortKey],
    skip: Option<usize>,
    limit: Option<usize>,
) -> Batch {
    let n = batch.rows();
    let mut idx: Vec<usize> = (0..n).collect();
    if !keys.is_empty() {
        let key_cols: Vec<Col> = keys.iter().map(|k| eval(&k.expr, store, batch)).collect();
        // Stable sort: equal keys keep input order, so the last key's ties fall
        // back to arrival order deterministically.
        idx.sort_by(|&a, &b| {
            for (kc, k) in key_cols.iter().zip(keys) {
                let ord = value::cmp_total(&kc.value_at(a), &kc.value_at(b));
                let ord = if k.descending { ord.reverse() } else { ord };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });
    }
    let start = skip.unwrap_or(0).min(idx.len());
    let end = limit.map_or(idx.len(), |l| start.saturating_add(l).min(idx.len()));
    batch.gather(&idx[start..end])
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
fn aggregate(batch: &Batch, store: &Store, keys: &[(String, Expr)], aggs: &[Agg]) -> Batch {
    let n = batch.rows();
    let key_cols: Vec<Col> = keys.iter().map(|(_, e)| eval(e, store, batch)).collect();

    // With no keys the whole input is one group, and a scalar aggregate over
    // EMPTY input still emits that one group (SQL: `count(*)` over nothing is 0,
    // one row) — hence n_groups is forced to 1 there. A GROUPED aggregate over
    // empty input has no groups and emits zero rows, which falls out naturally.
    let (group_of, first_row, n_groups) = if keys.is_empty() {
        (vec![0u32; n], Vec::new(), 1)
    } else {
        let (g, fr) = assign_groups(&key_cols, n);
        let ng = fr.len();
        (g, fr, ng)
    };

    // Key output columns: each key's value at its group's first row.
    let mut slots: Vec<Col> = if keys.is_empty() {
        Vec::new()
    } else {
        key_cols.iter().map(|c| c.gather(&first_row)).collect()
    };

    for agg in aggs {
        let arg_col = agg.arg.as_ref().map(|e| eval(e, store, batch));
        slots.push(Col::Gen(fold_grouped(
            agg,
            arg_col.as_ref(),
            &group_of,
            n_groups,
        )));
    }

    Batch::of(slots)
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
            Col::Nodes(v) => return group_by(n, v.iter().map(|&x| u64::from(x))),
            Col::Bool(v) => return group_by(n, v.iter().map(|&b| u64::from(b))),
            Col::Str(v) => return group_by_ref(n, v.iter().map(std::convert::AsRef::as_ref)),
            Col::Gen(_) => {} // mixed: fall through to the byte-key path
        }
    }
    // General path: self-delimiting byte key per row, reused buffer, allocate
    // only when a row opens a new group.
    let mut of: HashMap<Vec<u8>, u32> = HashMap::new();
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
    let mut of: HashMap<K, u32> = HashMap::new();
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

/// Group by a borrowed key (strings): the owned key is cloned only when a row
/// opens a new group, so a million-row column over a thousand names clones a
/// thousand `Arc`s, not a million.
fn group_by_ref<'a>(n: usize, keys: impl Iterator<Item = &'a str>) -> (Vec<u32>, Vec<usize>) {
    let mut of: HashMap<Box<str>, u32> = HashMap::new();
    let mut group_of = Vec::with_capacity(n);
    let mut first_row = Vec::new();
    for (i, k) in keys.enumerate() {
        let g = match of.get(k) {
            Some(&g) => g,
            None => {
                let g = first_row.len() as u32;
                of.insert(Box::from(k), g);
                first_row.push(i);
                g
            }
        };
        group_of.push(g);
    }
    (group_of, first_row)
}

/// Fold one aggregate to one value per group in a single streaming pass over the
/// group labelling. Null policy and ordering come from the value contract;
/// nothing here restates them.
fn fold_grouped(agg: &Agg, arg_col: Option<&Col>, group_of: &[u32], n_groups: usize) -> Vec<Value> {
    // `count(*)` — no argument — is each group's row count: a pure tally.
    if agg.func == AggFn::Count && agg.arg.is_none() {
        let mut tally = vec![0f64; n_groups];
        for &g in group_of {
            tally[g as usize] += 1.0;
        }
        return tally.into_iter().map(Value::Num).collect();
    }
    let Some(col) = arg_col else {
        return vec![Value::Null; n_groups]; // sum/min/max/avg with no argument
    };

    match agg.func {
        AggFn::Count if agg.distinct => {
            // Per-group distinct count. A dedicated set per group, keyed by the
            // grouping bytes; a group entry is allocated only for a new value.
            let mut sets: Vec<std::collections::HashSet<Vec<u8>>> = (0..n_groups)
                .map(|_| std::collections::HashSet::new())
                .collect();
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
            // total + count of non-null NUMERIC values; a non-null non-numeric
            // poisons its group to NULL (never coerced to NaN), and an all-null
            // (or empty) group is NULL — only count is 0 over nothing.
            let mut total = vec![0f64; n_groups];
            let mut cnt = vec![0u64; n_groups];
            let mut poison = vec![false; n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                match col.value_at(i) {
                    Value::Null => {}
                    Value::Num(x) => {
                        total[g as usize] += x;
                        cnt[g as usize] += 1;
                    }
                    _ => poison[g as usize] = true,
                }
            }
            (0..n_groups)
                .map(|g| {
                    if poison[g] || cnt[g] == 0 {
                        Value::Null
                    } else if agg.func == AggFn::Sum {
                        Value::Num(total[g])
                    } else {
                        Value::Num(total[g] / cnt[g] as f64)
                    }
                })
                .collect()
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
    }
}

/// A hop: for each input row, expand the node in slot `from` along `dir`,
/// filtered by `edge_label`; emit one output row per matching neighbour with the
/// existing slots replicated and the neighbour appended as a new slot. This is
/// the bulk (lineage-free) strategy: `keep` records which input row each output
/// row came from, `nbrs` the landed node — the existing slots are gathered by
/// `keep`, so no per-row struct is built.
fn expand(batch: &Batch, store: &Store, from: usize, dir: Dir, edge_label: Option<&str>) -> Batch {
    // An empty expand still appends the landed slot (all rows dropped), so the
    // output has K+1 slots exactly as a successful expand would — a projection
    // referencing the new slot must not go out of bounds.
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        let mut b = Batch::of(slots);
        if batch.lineage.is_some() {
            b.lineage = Some(Lineage::empty());
        }
        b
    };
    // Resolve the edge label to an interned id up front; an unknown label matches
    // nothing (not everything).
    let want: Option<u32> = match edge_label {
        None => None,
        Some(name) => match store.etype_id(name) {
            Some(id) => Some(id),
            None => return empty(),
        },
    };
    let Col::Nodes(src) = batch.slot(from) else {
        // Only a node frontier can be expanded; anything else yields nothing.
        return empty();
    };

    let type_ok = |et: u32| want.is_none_or(|w| w == et);
    let mut keep = Vec::new();
    let mut nbrs = Vec::new();
    for (row, &v) in src.iter().enumerate() {
        let out = matches!(dir, Dir::Out | Dir::Both);
        let inc = matches!(dir, Dir::In | Dir::Both);
        if out {
            for a in store.out(v) {
                if type_ok(a.etype) {
                    keep.push(row);
                    nbrs.push(a.nbr);
                }
            }
        }
        if inc {
            for a in store.inc(v) {
                if type_ok(a.etype) {
                    keep.push(row);
                    nbrs.push(a.nbr);
                }
            }
        }
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(nbrs.clone()));
    let mut out = Batch::of(slots);
    // Lineage strategy: when the input carried a path, extend each output row's
    // path by the neighbour it landed on. This is the ONLY place Expand differs
    // for lineage — the frontier work above is identical.
    if let Some(lin) = &batch.lineage {
        out.lineage = Some(lin.extend(&keep, &nbrs));
    }
    out
}

/// A quantified hop: for each input row, enumerate every path of length in
/// `min..=max` from the node in `from`, and emit one output row per path with the
/// reached endpoint appended as a new slot. `min == 0` emits the source itself.
///
/// `trail` chooses the semantics and nothing else does: true forbids reusing an
/// edge within one path (a trail), false allows it (a walk). They diverge on a
/// cycle/self-loop — pinned by the tests — and are never conflated with a chain
/// of separate fixed `Expand`s (which is always a walk).
#[allow(clippy::too_many_arguments)]
fn var_length(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: Option<&str>,
    min: u32,
    max: u32,
    trail: bool,
) -> Batch {
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        Batch::of(slots)
    };
    let want: Option<u32> = match edge_label {
        None => None,
        Some(name) => match store.etype_id(name) {
            Some(id) => Some(id),
            None => return empty(),
        },
    };
    let Col::Nodes(src) = batch.slot(from) else {
        return empty();
    };

    let mut keep = Vec::new();
    let mut ends = Vec::new();
    let mut used: Vec<u32> = Vec::new(); // edge ids on the current path (trail only)
    for (row, &v) in src.iter().enumerate() {
        varlen_dfs(
            store, v, 0, min, max, dir, want, trail, &mut used, row, &mut keep, &mut ends,
        );
        debug_assert!(used.is_empty());
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(ends));
    Batch::of(slots)
}

/// Depth-first path enumeration for `var_length`. Emits `(row, endpoint)` at
/// every length in `min..=max` reached from the source, pushing straight into
/// `keep`/`ends` (a recursion-friendly alternative to a closure). For a trail,
/// `used` holds the edge ids on the current path and blocks reuse.
#[allow(clippy::too_many_arguments)]
fn varlen_dfs(
    store: &Store,
    v: u32,
    len: u32,
    min: u32,
    max: u32,
    dir: Dir,
    want: Option<u32>,
    trail: bool,
    used: &mut Vec<u32>,
    row: usize,
    keep: &mut Vec<usize>,
    ends: &mut Vec<u32>,
) {
    if len >= min {
        keep.push(row);
        ends.push(v);
    }
    if len == max {
        return;
    }
    let mut adjs: Vec<crate::store::Adj> = Vec::new();
    if matches!(dir, Dir::Out | Dir::Both) {
        adjs.extend_from_slice(store.out(v));
    }
    if matches!(dir, Dir::In | Dir::Both) {
        adjs.extend_from_slice(store.inc(v));
    }
    for a in adjs {
        if want.is_some_and(|w| w != a.etype) {
            continue;
        }
        if trail && used.contains(&a.eid) {
            continue; // a trail may not reuse an edge
        }
        if trail {
            used.push(a.eid);
        }
        varlen_dfs(
            store,
            a.nbr,
            len + 1,
            min,
            max,
            dir,
            want,
            trail,
            used,
            row,
            keep,
            ends,
        );
        if trail {
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
    edge_label: Option<&str>,
    max: Option<u32>,
) -> Batch {
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        Batch::of(slots)
    };
    let want: Option<u32> = match edge_label {
        None => None,
        Some(name) => match store.etype_id(name) {
            Some(id) => Some(id),
            None => return empty(),
        },
    };
    let Col::Nodes(src) = batch.slot(from) else {
        return empty();
    };

    let mut keep = Vec::new();
    let mut ends = Vec::new();
    for (row, &start) in src.iter().enumerate() {
        let mut visited = std::collections::HashSet::new();
        visited.insert(start);
        let mut q: VecDeque<(u32, u32)> = VecDeque::new();
        q.push_back((start, 0));
        while let Some((v, d)) = q.pop_front() {
            if max.is_some_and(|m| d >= m) {
                continue; // reached the hop cap; do not expand further
            }
            let mut adjs: Vec<crate::store::Adj> = Vec::new();
            if matches!(dir, Dir::Out | Dir::Both) {
                adjs.extend_from_slice(store.out(v));
            }
            if matches!(dir, Dir::In | Dir::Both) {
                adjs.extend_from_slice(store.inc(v));
            }
            for a in adjs {
                if want.is_some_and(|w| w != a.etype) {
                    continue;
                }
                // First reach = shortest reach (BFS), and only the first is kept.
                if visited.insert(a.nbr) {
                    keep.push(row);
                    ends.push(a.nbr);
                    q.push_back((a.nbr, d + 1));
                }
            }
        }
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(ends));
    Batch::of(slots)
}

/// Evaluate `expr` over every row of `batch`, producing a column.
fn eval(expr: &Expr, store: &Store, batch: &Batch) -> Col {
    match expr {
        Expr::Slot(n) => batch.slot(*n).clone(),
        Expr::Lit(v) => broadcast(v.clone(), batch.rows()),
        Expr::Prop { slot, key } => read_property(store, batch.slot(*slot), key),
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
        Expr::Not(inner) => {
            let c = eval(inner, store, batch);
            map_bool(&c, |b| b.map(|x| !x))
        }
        Expr::And(l, r) => zip_bool(store, batch, l, r, |a, b| match (a, b) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        }),
        Expr::Or(l, r) => zip_bool(store, batch, l, r, |a, b| match (a, b) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        }),
        Expr::Compare { op, left, right } => {
            let l = eval(left, store, batch);
            let r = eval(right, store, batch);
            compare(*op, &l, &r)
        }
    }
}

/// Read `key` off an element frontier as a column, bulk-gathering the typed
/// storage column and staying unboxed when it and every read entry are
/// present-and-typed; fall to `Gen` (with nulls) otherwise.
fn read_property(store: &Store, col: &Col, key: &str) -> Col {
    let Col::Nodes(ids) = col else {
        return Col::Gen(vec![Value::Null; col.len()]);
    };
    let Some(column) = store.column(key) else {
        return Col::Gen(vec![Value::Null; ids.len()]);
    };
    match column {
        Column::Num { data, present } if ids.iter().all(|&i| present[i as usize]) => {
            Col::Num(ids.iter().map(|&i| data[i as usize]).collect())
        }
        Column::Str { data, present } if ids.iter().all(|&i| present[i as usize]) => {
            Col::Str(ids.iter().map(|&i| data[i as usize].clone()).collect())
        }
        Column::Bool { data, present } if ids.iter().all(|&i| present[i as usize]) => {
            Col::Bool(ids.iter().map(|&i| data[i as usize]).collect())
        }
        _ => Col::Gen(ids.iter().map(|&i| store.prop(i, key)).collect()),
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
        let res = match op {
            CompareOp::Eq => value::equals(&a, &b),
            CompareOp::Ne => !value::equals(&a, &b),
            CompareOp::Lt => value::cmp_total(&a, &b).is_lt(),
            CompareOp::Le => value::cmp_total(&a, &b).is_le(),
            CompareOp::Gt => value::cmp_total(&a, &b).is_gt(),
            CompareOp::Ge => value::cmp_total(&a, &b).is_ge(),
        };
        out.push(Some(res));
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
fn as_truth(col: &Col) -> Vec<Option<bool>> {
    match col {
        Col::Bool(bs) => bs.iter().map(|&b| Some(b)).collect(),
        other => (0..other.len())
            .map(|i| match other.value_at(i) {
                Value::Bool(b) => Some(b),
                _ => None,
            })
            .collect(),
    }
}

fn map_bool(col: &Col, f: impl Fn(Option<bool>) -> Option<bool>) -> Col {
    truth_to_col(as_truth(col).into_iter().map(f).collect())
}

fn zip_bool(
    store: &Store,
    batch: &Batch,
    l: &Expr,
    r: &Expr,
    f: impl Fn(Option<bool>, Option<bool>) -> Option<bool>,
) -> Col {
    let lc = as_truth(&eval(l, store, batch));
    let rc = as_truth(&eval(r, store, batch));
    let n = lc.len().min(rc.len());
    truth_to_col((0..n).map(|i| f(lc[i], rc[i])).collect())
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

    // --- Expand ---

    /// `MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name` — two slots bound,
    /// row per matching edge.
    #[test]
    fn expand_binds_both_ends() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
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
            .expand(0, Dir::Out, Some("WORKS_ON"))
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["graphdb"]);
    }

    /// Filtering on the FAR end after an expand — the far slot's property.
    #[test]
    fn filter_on_the_expanded_end() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
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
            .expand(0, Dir::In, Some("KNOWS"))
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
            .expand(0, Dir::Out, Some("NOPE"))
            .project(vec![("x".into(), prop(1, "name"))]);
        assert_eq!(run(&plan, &store).rows.len(), 0);
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

    /// sum over an empty group is NULL, not 0 — only count is 0 over nothing.
    #[test]
    fn sum_over_empty_is_null_count_is_zero() {
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
            ],
        );
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1); // scalar aggregate still emits one row
        assert_eq!(num(&out.rows[0][0]), 0.0); // count(*) = 0
        assert!(out.rows[0][1].is_null()); // sum = NULL
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
        let plan = scan("Person").expand(0, Dir::Out, Some("KNOWS")).aggregate(
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

    // --- Order + Page ---

    fn asc(slot: usize, key: &str) -> crate::ir::SortKey {
        crate::ir::SortKey {
            expr: prop(slot, key),
            descending: false,
        }
    }
    fn desc(slot: usize, key: &str) -> crate::ir::SortKey {
        crate::ir::SortKey {
            expr: prop(slot, key),
            descending: true,
        }
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
    #[test]
    fn distinct_reached_set() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .project(vec![("who".into(), prop(1, "name"))])
            .distinct();
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["bob", "carol"]);
    }

    // --- Join (multi-pattern / shared variable) ---

    /// `MATCH (a)-[:KNOWS]->(b), (a)-[:WORKS_ON]->(c)` sharing `a`. Left slots
    /// [a,b], right slots [a,c]; join on left a (0) == right a (0); output slots
    /// [a, b, a', c]. Only alice has a WORKS_ON, so only her KNOWS rows survive.
    #[test]
    fn join_shared_start_variable() {
        let store = social();
        let left = scan("Person").expand(0, Dir::Out, Some("KNOWS"));
        let right = scan("Person").expand(0, Dir::Out, Some("WORKS_ON"));
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
        let left = scan("P").expand(0, Dir::Out, Some("R"));
        let right = scan("P").expand(0, Dir::Out, Some("S"));
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
        let left = scan("Person").expand(0, Dir::Out, Some("KNOWS"));
        let right = scan("Person").expand(0, Dir::Out, Some("WORKS_ON"));
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
            .var_length(0, Dir::Out, Some("R"), 1, 2, true)
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
            .var_length(0, Dir::Out, Some("R"), 0, 2, true)
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
            .var_length(0, Dir::Out, Some("R"), 1, 2, false)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&walk, &store).rows.len(), 2, "walk reuses the edge");

        let trail = base
            .var_length(0, Dir::Out, Some("R"), 1, 2, true)
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
            .var_length(0, Dir::Out, Some("R"), 1, 3, true)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&trail, &store).rows.len(), 2); // b (len1), a (len2)

        let walk = from_a
            .var_length(0, Dir::Out, Some("R"), 1, 3, false)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&walk, &store).rows.len(), 3); // b, a, b
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
            .var_length(0, Dir::Out, Some("R"), 2, 2, true)
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
            .shortest_path(0, Dir::Out, Some("R"), None)
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
            .shortest_path(0, Dir::Out, Some("R"), None)
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
            .shortest_path(0, Dir::Out, Some("R"), Some(2))
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["b", "c"]); // d (distance 3) beyond the cap
    }

    /// A cycle does not loop forever — each node is reached once.
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
            .shortest_path(0, Dir::Out, Some("R"), None)
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        // b(1), c(2); a is the source, not re-emitted despite the cycle back.
        assert_eq!(got, vec!["b", "c"]);
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
            .expand(0, Dir::Out, Some("R"))
            .expand(1, Dir::Out, Some("R"))
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

    /// A one-hop path is two nodes; the source's own path (length-0 walk via a
    /// bare scan) is one node.
    #[test]
    fn path_length_grows_with_hops() {
        let store = social();
        // alice -KNOWS-> {bob, carol}; RETURN path per edge.
        let plan = scan("Person")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))))
            .expand(0, Dir::Out, Some("KNOWS"))
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
            .expand(0, Dir::Out, Some("KNOWS"))
            .project(vec![("b".into(), prop(1, "name"))]);
        assert!(!super::needs_lineage(&plain), "no Path read -> no lineage");
        // The pulled batch (before the lineage-dropping Project) has no sidecar.
        let inner = scan("Person").expand(0, Dir::Out, Some("KNOWS"));
        assert!(super::pull(&inner, &store, false).lineage.is_none());

        let with_path = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .project(vec![("p".into(), Expr::Path)]);
        assert!(super::needs_lineage(&with_path), "Path read -> lineage");
        // With track=true the expand carries a sidecar.
        assert!(super::pull(&inner, &store, true).lineage.is_some());
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
            .expand(0, Dir::Out, Some("R"))
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
