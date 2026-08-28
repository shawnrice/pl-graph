use super::varlen::{run_varlen, DistinctEndpointEmit};
use super::*;
use crate::batch::{Batch, Col};
use crate::ir::Expr;
use crate::store::{Column, Store};
use crate::value::{self, Value};

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
                    Value::Str(dict[codes[i] as usize].clone().into())
                } else {
                    Value::Null
                }
            }
            Self::Str { data, present } => {
                if present[i] {
                    Value::Str(data[i].clone().into())
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
pub(super) fn try_distinct_scan_multi(input: &Plan, store: &Store) -> Option<Batch> {
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
pub(super) fn distinct_batch(batch: Batch) -> Batch {
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
pub(super) fn try_distinct_varlen_expr(input: &Plan, store: &Store) -> Option<Batch> {
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
pub(super) fn try_distinct_frontier_prop(input: &Plan, store: &Store) -> Option<Batch> {
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
                        out.push(Value::Str(data[i].clone().into()));
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
                        out.push(Value::Str(dict[c].clone().into()));
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
pub(super) fn try_distinct_identical_cols(
    input: &Plan,
    store: &Store,
    track: bool,
) -> Option<Batch> {
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

pub(super) fn try_distinct_frontier_multi(input: &Plan, store: &Store) -> Option<Batch> {
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
pub(super) fn try_distinct_scan_prop(input: &Plan, store: &Store) -> Option<Batch> {
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
                        out.push(Value::Str(data[i].clone().into()));
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
                        out.push(Value::Str(dict[c].clone().into()));
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
