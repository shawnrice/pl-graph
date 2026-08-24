use super::*;
use crate::batch::{Batch, Col};
use crate::ir::Expr;
use crate::store::{Column, Store};
use crate::value::{self, Value};

/// Keep the rows of `batch` whose key (the `key_slots` tuple) is seen for the FIRST
/// time, threading the seen-set across calls (streamed dedup). A SINGLE node/edge key
/// slot (`typed`) deduplicates by raw `u32` id — no per-row byte-key serialization,
/// which dominated `dedup()` over a big fan-out; `value_at(Col::Nodes)` is `Node(id)`
/// so a node's group key IS its id, keeping it byte-identical.
pub(super) fn distinct_by_keep(
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
pub(super) fn distinct_by_typed(batch: &Batch, key_slots: &[usize]) -> bool {
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
pub(super) fn try_distinct_by_streamed(
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
    Ok(Some(concat_batches(&acc, store)))
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
pub(super) fn try_scan_top_k(
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
pub(super) fn try_late_materialize(
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
    if start == 0 && end >= n {
        // The window is the WHOLE set (no skip, and the limit reaches the end) — a full
        // projection is unavoidable; nothing to late-materialize. With a SKIP, though, the
        // prefix `[0, start)` is dropped, so the window is NOT the whole set: late-
        // materialize still projects only `[start, end)`, which also means a fallible
        // projection never runs for a paged-out row (the reason to keep this path here).
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
pub(super) fn order_page(
    batch: &Batch,
    store: &Store,
    keys: &[crate::ir::SortKey],
    skip: Option<usize>,
    limit: Option<usize>,
    fault_on_element: bool,
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
        // A sort KEY that resolves to a graph element (a vertex/edge) has no natural order.
        // Gremlin `order()` FAULTS (TinkerPop throws) — the runtime backstop for a MIXED/branch
        // frontier the parser could not classify (`both(...).inject(1).order()`). GQL `ORDER BY
        // <element>` instead sorts by the element's EXTERNAL ID (like Cypher / the pure-TS
        // `@lenke/gql`), so `CALL algo() YIELD node … ORDER BY node` orders by id, not faults.
        let key_cols = if key_cols.iter().any(col_has_element) {
            if fault_on_element {
                return Err(
                    "order() over graph elements is not supported — elements have no \
                            natural order; use order().by('<key>')"
                        .into(),
                );
            }
            key_cols
                .into_iter()
                .map(|c| element_col_to_ext_id(c, store))
                .collect()
        } else {
            key_cols
        };
        let key_cols = typed_key_cols(key_cols);
        sort_idx(&mut idx, &key_cols, keys, end);
    }
    Ok(batch.gather(&idx[start..end]))
}

/// Replace each graph-element cell in a sort-key column with its EXTERNAL ID (a string),
/// so a GQL `ORDER BY <element>` sorts by id — matching the pure-TS engine, which orders
/// nodes/edges by their id lexicographically. A missing external id falls back to the dense
/// id as a number (defensive; live elements always have one). Non-element columns pass through.
fn element_col_to_ext_id(col: Col, store: &Store) -> Col {
    let node_key = |id: u32| {
        store
            .node_ext_id(id)
            .map_or(Value::Num(f64::from(id)), Value::Str)
    };
    let edge_key = |eid: u32| {
        store
            .edge_ext_id(eid)
            .map_or(Value::Num(f64::from(eid)), Value::Str)
    };
    match col {
        Col::Nodes(ids) => Col::Gen(ids.iter().map(|&id| node_key(id)).collect()),
        Col::Edges(eids) => Col::Gen(eids.iter().map(|&eid| edge_key(eid)).collect()),
        Col::Gen(vs) => Col::Gen(
            vs.into_iter()
                .map(|v| match v {
                    Value::Node(id) => node_key(id),
                    Value::Edge(eid) => edge_key(eid),
                    other => other,
                })
                .collect(),
        ),
        other => other,
    }
}

/// A sort-key column that carries a graph element — a `Nodes`/`Edges` slot or a `Gen`
/// column holding any `Value::Node`/`Value::Edge`. Elements have no natural order, so
/// ordering by one faults (see [`order_page`]).
/// The `otherV` reference vertex for an edge `(src, dst)` given the row's node path: the vertex
/// the traverser ARRIVED from, i.e. the node just before the edge. When the path's last node is
/// the edge's own landed endpoint (an `outE`/`inE` recorded it), the arrival is the node before
/// it; otherwise the last node is the arrival. `None` for an edge with no prior vertex.
pub(super) fn otherv_reference(nodes: &[Value], src: u32, dst: u32) -> Option<u32> {
    match nodes.last().map(num_as_u32) {
        Some(last) if (last == src || last == dst) && nodes.len() >= 2 => {
            Some(num_as_u32(&nodes[nodes.len() - 2]))
        }
        other => other,
    }
}

fn col_has_element(col: &Col) -> bool {
    match col {
        Col::Nodes(_) | Col::Edges(_) => true,
        Col::Gen(vs) => vs
            .iter()
            .any(|v| matches!(v, Value::Node(_) | Value::Edge(_))),
        _ => false,
    }
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
pub(super) const FRONTIER_FOLD_MIN: usize = 50_000;

/// First-seen grouping for a SINGLE node-PROPERTY key over a materialized batch, via the
/// typed [`frontier_group_by`] (Str/Num/Bool/Dict read off storage) — the general-batch
/// analogue of [`try_dict_grouping`]. Unlike the chain-only `try_frontier_group_fold`, it
/// works on whatever batch the aggregate already pulled (a join, a filtered frontier), so a
/// grouped aggregate over ANY shape skips the `Col::Str` + byte-key `assign_groups` pays.
/// `None` unless the sole key is `Prop{slot, <plain col>}` over a `Col::Nodes` slot backed
/// by a Str/Num/Bool/Dict column.
pub(super) fn try_node_prop_grouping(
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
