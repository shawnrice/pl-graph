use super::evaluator::*;
use super::order::FRONTIER_FOLD_MIN;
use super::*;
use crate::batch::{Batch, Col};
use crate::gstr::GStr;
use crate::ir::{Agg, AggFn};
use crate::store::{Column, Store};
use crate::value::{self, Value};

pub(super) fn aggregate(
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
        // A NUMERIC aggregate over a raw ELEMENT column (a vertex/edge frontier — e.g. a
        // union/coalesce/optional arm that yields elements, whose 'unknown' frontier slips
        // past the Gremlin parser's static current_is_element guard) is a type error: a
        // Vertex/Edge is not a number. `value_at` surfaces a node as its dense id, which
        // would SILENTLY SUM THE IDS; TinkerPop throws ClassCastException and pure-TS faults,
        // so fault here to match (`g.V().union(out(), …).sum()`). Collect/fold over elements
        // is fine (it builds a list of element maps), so only the numeric reducers fault.
        // A graph ELEMENT (a vertex/edge, carried unboxed in a Gen column) is neither numeric
        // nor comparable, so it faults for EVERY numeric reducer (min/max included).
        let elem_col = match arg_col.as_ref() {
            Some(Col::Nodes(v)) => !v.is_empty(),
            Some(Col::Edges(v)) => !v.is_empty(),
            Some(Col::Gen(vs)) => vs
                .iter()
                .any(|v| matches!(v, Value::Node(_) | Value::Edge(_))),
            _ => false,
        };
        // A path/list is not a NUMBER, so `sum`/`mean` fault (`union(path(), …).sum()`), but it
        // IS ordered (a list has a total order), so `min`/`max` over lists is fine (GQL
        // `min(m.tags)`) — only the strictly-numeric reducers reject it.
        let list_col = matches!(arg_col.as_ref(), Some(Col::Gen(vs))
            if vs.iter().any(|v| matches!(v, Value::List(_))));
        let numeric_only = matches!(
            agg.func,
            AggFn::Sum | AggFn::Avg | AggFn::StddevPop | AggFn::StddevSamp
        );
        let numeric_reducer = numeric_only
            || matches!(
                agg.func,
                AggFn::Min | AggFn::Max | AggFn::PercentileCont | AggFn::PercentileDisc
            );
        if (numeric_reducer && elem_col) || (numeric_only && list_col) {
            return Err(format!(
                "{}() over graph elements is not supported — a vertex/edge is not a number; \
                 project with values('<key>') first",
                agg.name
            ));
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
pub(super) fn frontier_group_by(
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
pub(super) fn try_frontier_group_fold(
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
pub(super) fn assign_groups(key_cols: &[Col], n: usize) -> (Vec<u32>, Vec<usize>) {
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
fn group_by_arc(keys: &[GStr]) -> (Vec<u32>, Vec<usize>) {
    // Pre-size for the worst case (all-distinct) so the map never rehashes while
    // filling — the rehash chain dominated an all-unique million-key merge.
    let mut of: FnvMap<GStr, u32> =
        FnvMap::with_capacity_and_hasher(keys.len(), Default::default());
    let mut group_of = Vec::with_capacity(keys.len());
    let mut first_row = Vec::new();
    for (i, k) in keys.iter().enumerate() {
        let g = match of.get(k.as_ref()) {
            Some(&g) => g,
            None => {
                let g = first_row.len() as u32;
                of.insert(k.clone(), g);
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
pub(super) fn sort_local_cell(v: Value, descending: bool, by_key: bool) -> Value {
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
            use crate::temporal::{Duration, Temporal};
            // total + count of non-null NUMERIC values. SUM ALSO folds a group of DURATIONs
            // component-wise (ISO-8601 duration addition is always well-defined — months,
            // days, seconds and nanos add independently; only an i64 overflow faults). AVG
            // cannot: dividing months by an arbitrary count is ill-defined, so a duration
            // under AVG stays a DATA EXCEPTION. A group that MIXES numbers and durations, a
            // non-DURATION temporal (date/time), or any other non-null non-numeric value is
            // a data exception — sum()/avg() never coerce (the SQL rule of binary
            // arithmetic). NULLs are skipped. SUM of an empty/all-null group is 0, AVG NULL.
            let mut total = vec![0f64; n_groups];
            let mut cnt = vec![0u64; n_groups];
            let mut dur: Vec<Option<Duration>> = vec![None; n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                let g = g as usize;
                match col.value_at(i) {
                    Value::Null => {}
                    Value::Num(x) => {
                        if dur[g].is_some() {
                            return Err("sum() cannot mix numbers and durations".into());
                        }
                        total[g] += x;
                        cnt[g] += 1;
                    }
                    Value::Temporal(Temporal::Duration(d)) if agg.func == AggFn::Sum => {
                        if cnt[g] > 0 {
                            return Err("sum() cannot mix numbers and durations".into());
                        }
                        dur[g] = Some(match dur[g] {
                            Some(acc) => acc
                                .add(&d)
                                .ok_or_else(|| "duration sum is out of range".to_string())?,
                            None => d,
                        });
                    }
                    // A Gremlin multi-key `values('v','k')` arg is a LIST — flatten it,
                    // summing each numeric element (skipping non-numeric/null).
                    Value::List(items) if agg.null_on_empty => {
                        for el in &items {
                            if let Value::Num(x) = el {
                                total[g] += x;
                                cnt[g] += 1;
                            }
                        }
                    }
                    // A non-numeric SCALAR (a vertex/edge/string) is a data exception in
                    // BOTH engines — sum()/avg() never coerce. TinkerPop's `sum()` over
                    // vertices throws (you can't add a Vertex), so `g.V().sum()` faults
                    // rather than silently skipping. (Gremlin still differs only in that an
                    // EMPTY/all-null sum() is NULL — handled below — not GQL/SQL's 0.)
                    _ => return Err("sum()/avg() require numeric values".into()),
                }
            }
            (0..n_groups)
                .map(|g| {
                    if let Some(d) = dur[g] {
                        Value::Temporal(Temporal::Duration(d)) // SUM of durations
                    } else if agg.func == AggFn::Sum {
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
            // min()/max() order WITHIN one type — TinkerPop compares pure numbers or pure
            // strings fine (`max('a','b')` is 'b', verified against gremlin-console) — but
            // a CROSS-TYPE comparison (a number and a string in the SAME group) is a
            // ClassCastException there. Gremlin (`numeric_only`) faults on such a mixed
            // group; GQL (`numeric_only` false) keeps the total order. NULLs are skipped;
            // NaN stays a number (cmp_total: NaN greatest).
            let want_min = agg.func == AggFn::Min;
            let mut best: Vec<Option<Value>> = vec![None; n_groups];
            // Each group's established "is this a numeric group?" from its first non-null
            // value; a later value of the other kind is the cross-type fault.
            let mut is_num_group: Vec<Option<bool>> = vec![None; n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                let v = col.value_at(i);
                if v.is_null() {
                    continue;
                }
                if agg.numeric_only {
                    let is_num = matches!(v, Value::Num(_));
                    match is_num_group[g as usize] {
                        None => is_num_group[g as usize] = Some(is_num),
                        Some(prev) if prev != is_num => {
                            return Err("min()/max() cannot compare across types \
                                        (a number and a non-number in the same group)"
                                .into());
                        }
                        _ => {}
                    }
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
