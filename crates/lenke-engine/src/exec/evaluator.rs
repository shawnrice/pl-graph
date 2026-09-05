use super::scalar::*;
use super::*;
use crate::batch::{Batch, Col};
use crate::gstr::GStr;
use crate::ir::Expr;
use crate::store::{Column, Store};
use crate::value::{self, Value};

/// Elementwise `l OP r` over two already-evaluated columns — the general arithmetic
/// body, shared by `Expr::Arith` and its scalar fast path's non-numeric fallback.
/// Raw f64 when both are `Col::Num`; otherwise per-cell via the value contract (a
/// NULL / non-numeric operand → NULL, a temporal operand → `temporal_arith`). Div/Rem
/// by a zero divisor (the RIGHT operand) throws, matching the TS engine's DataException.
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
                // is the TS engine's SQL-style rule (`'abc' + 1` throws; `1 + null` is null).
                _ if a.is_null() || b.is_null() => Value::Null,
                _ => return Err("arithmetic requires a number".into()),
            }
        };
        out.push(v);
    }
    Ok(Col::Gen(out))
}

/// Evaluate `expr` over every row of `batch`, producing a column.
/// Lazy (SQL-standard) CASE evaluation — the fallback when the inline eager path throws,
/// because a type error may live in a branch a row never actually reaches. Each condition
/// is evaluated only over the rows still UNRESOLVED at that branch, and each value only
/// over the rows that select it, via a gathered sub-batch. So a branch's type error
/// surfaces ONLY if a row that genuinely reaches that condition (or takes that value) is
/// ill-typed — matching TS and SQL, where `CASE WHEN c THEN safe ELSE risky END` never
/// evaluates `risky` for a row where `c` holds. Slower (per-branch gather), so it runs only
/// on the error path; the eager path stays fully vectorized.
fn eval_case_masked(
    branches: &[(Expr, Expr)],
    otherwise: Option<&Expr>,
    store: &Store,
    batch: &Batch,
) -> Result<Col, String> {
    let n = batch.rows();
    let mut out = vec![Value::Null; n];
    // Original row indices not yet resolved by an earlier branch.
    let mut pending: Vec<usize> = (0..n).collect();
    for (cond, val) in branches {
        if pending.is_empty() {
            break;
        }
        // The condition, evaluated ONLY over the rows that still reach this branch.
        let cond_col = eval(cond, store, &batch.gather(&pending))?;
        let mut taken: Vec<usize> = Vec::new();
        let mut still: Vec<usize> = Vec::with_capacity(pending.len());
        for (k, &orig) in pending.iter().enumerate() {
            match cond_col.value_at(k) {
                Value::Bool(true) => taken.push(orig),
                Value::Bool(false) | Value::Null => still.push(orig),
                _ => return Err(TRUTH_TYPE_ERR.to_string()),
            }
        }
        if !taken.is_empty() {
            // The value, evaluated ONLY over the rows that took this branch.
            let vcol = eval(val, store, &batch.gather(&taken))?;
            for (k, &orig) in taken.iter().enumerate() {
                out[orig] = vcol.value_at(k);
            }
        }
        pending = still;
    }
    // Rows that matched no branch → ELSE (over just those rows) or NULL.
    if !pending.is_empty() {
        if let Some(e) = otherwise {
            let ecol = eval(e, store, &batch.gather(&pending))?;
            for (k, &orig) in pending.iter().enumerate() {
                out[orig] = ecol.value_at(k);
            }
        }
    }
    Ok(Col::Gen(out))
}

pub(super) fn eval(expr: &Expr, store: &Store, batch: &Batch) -> Result<Col, String> {
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
        // negative / non-integer index → NULL; null-safe. Mirrors the TS engine.
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
                        .find(|(fk, _)| fk.as_ref() == k.as_str())
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
            // A bound path RETURNs as a rich Path object `{vertices, edges, length}`
            // (key order matches the pure-TS Path serialization), each vertex/edge a
            // full element map. NULL when the plan tracks no lineage (which
            // `needs_lineage` prevents when Path is actually read).
            Some(lin) => Col::Gen(
                (0..batch.rows())
                    .map(|i| {
                        let vertices = path_node_values(store, lin.path_at(i));
                        let edges = path_edge_values(store, lin.edges_at(i));
                        let len = edges.len() as f64;
                        Value::Map(std::sync::Arc::new(vec![
                            (Value::Str("vertices".into()), Value::List(vertices)),
                            (Value::Str("edges".into()), Value::List(edges)),
                            (Value::Str("length".into()), Value::Num(len)),
                        ]))
                    })
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
        Expr::GremlinFullPath { bys } => match &batch.lineage {
            Some(lin) => Col::Gen(
                (0..batch.rows())
                    .map(|i| {
                        let (svals, stags) = lin.steps_at(i);
                        let out: Vec<Value> = svals
                            .iter()
                            .zip(stags)
                            .enumerate()
                            .map(|(p, (v, &tag))| {
                                let by = if bys.is_empty() {
                                    &crate::ir::GPathBy::Element
                                } else {
                                    &bys[p % bys.len()]
                                };
                                match tag {
                                    crate::batch::STEP_NODE => {
                                        render_gpath_elem(store, num_as_u32(v), false, by)
                                    }
                                    crate::batch::STEP_EDGE => {
                                        render_gpath_elem(store, num_as_u32(v), true, by)
                                    }
                                    // A projected scalar is its own path element (no `by`).
                                    _ => v.clone(),
                                }
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
                                // `nodes(p)` / `edges(p)` materialize the full element
                                // maps (a vertex/edge object each), not bare ids.
                                PathPart::Nodes => Value::List(path_node_values(store, nodes)),
                                PathPart::Relationships => {
                                    Value::List(path_edge_values(store, edges))
                                }
                                // Hops == number of relationships.
                                PathPart::Length => Value::Num(edges.len() as f64),
                                // ISO cardinality of a path: every element (nodes + edges).
                                PathPart::Cardinality => {
                                    Value::Num((nodes.len() + edges.len()) as f64)
                                }
                                PathPart::Elements => {
                                    // n0, e0, n1, e1, …, nk — each a full element map.
                                    let ns = path_node_values(store, nodes);
                                    let es = path_edge_values(store, edges);
                                    let mut items = Vec::with_capacity(ns.len() + es.len());
                                    for (j, node) in ns.iter().enumerate() {
                                        items.push(node.clone());
                                        if let Some(e) = es.get(j) {
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
                        // A branch/mixed frontier carries elements UNBOXED in a Gen column.
                        Value::Node(id) => store.node_ext_id(id).map_or(Value::Null, Value::Str),
                        Value::Edge(e) => store.edge_ext_id(e).map_or(Value::Null, Value::Str),
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
                        Value::Edge(e) => store
                            .edge_type_name(e)
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
                        let mut labels: Vec<GStr> = Vec::with_capacity(ids.len());
                        let mut all_labelled = true;
                        for &id in ids {
                            match code_of.get(id as usize) {
                                Some(&c) if c != u32::MAX => {
                                    labels.push(names[c as usize].clone().into())
                                }
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
                                Some(&c) if c != u32::MAX => {
                                    Value::Str(names[c as usize].clone().into())
                                }
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
                                    Value::Str(arc.into())
                                }
                                None => Value::Null,
                            }
                        }
                        Value::Num(eid) if matches!(arg, Col::Edges(_)) => store
                            .edge_type_name(eid as u32)
                            .map_or(Value::Null, |t| Value::Str(t.into())),
                        // A branch/mixed frontier carries elements UNBOXED in a Gen column.
                        Value::Node(id) => store
                            .min_label_name(id)
                            .map_or(Value::Null, |nm| Value::Str(nm.into())),
                        Value::Edge(e) => store
                            .edge_type_name(e)
                            .map_or(Value::Null, |t| Value::Str(t.into())),
                        _ => Value::Null,
                    });
                }
                return Ok(Col::Gen(out));
            }
            // `element_map(element[, 'k1', …])` → Gremlin `elementMap()`: the TS engine's FLAT
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
                                        entries
                                            .push((Value::Str(Arc::clone(k).into()), col.read(ni)));
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
            // values (the TS engine's `propertyMap()`, not built here, is the list-wrapped
            // form). An optional trailing key list filters; no keys = every present
            // property. Keys are sorted (the engine's element-map convention; map key
            // order is set-based per policy). Gremlin-only — not in the GQL whitelist.
            if name == "value_map" || name == "property_map" {
                // `property_map` is `value_map` with each value wrapped in a single-
                // element LIST (a TinkerPop property is multi-valued; lenke is single).
                let wrap = name == "property_map";
                // A leading Bool arg (valueMap(true)) → also prepend id + label tokens,
                // never list-wrapped. Byte-identical to the json_out fast path's tokens.
                let tokens = args[1..]
                    .iter()
                    .any(|e| matches!(e, Expr::Lit(Value::Bool(true))));
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
                    .map(|i| match arg.value_at(i) {
                        Value::Num(id) if matches!(arg, Col::Nodes(_)) => {
                            let ni = id as usize;
                            let nid = id as u32;
                            let mut entries: Vec<(Value, Value)> = Vec::new();
                            if tokens {
                                entries.push((
                                    Value::Str(GStr::from("id")),
                                    store.node_ext_id(nid).map_or(Value::Null, Value::Str),
                                ));
                                entries.push((
                                    Value::Str(GStr::from("label")),
                                    store.labels_of(nid).first().map_or(Value::Null, |l| {
                                        Value::Str(GStr::from(l.as_str()))
                                    }),
                                ));
                            }
                            // node_cols are already sorted; the token pairs stay ahead.
                            entries.extend(
                                node_cols.iter().filter(|(_, col)| col.present_at(ni)).map(
                                    |(k, col)| {
                                        let v = col.read(ni);
                                        let v = if wrap { Value::List(vec![v]) } else { v };
                                        (Value::Str(Arc::clone(k).into()), v)
                                    },
                                ),
                            );
                            Value::Map(Arc::new(entries))
                        }
                        Value::Num(eid) if matches!(arg, Col::Edges(_)) => {
                            let eid = eid as u32;
                            let keys = if filter.is_empty() {
                                store.edge_prop_keys()
                            } else {
                                filter.clone()
                            };
                            let mut pairs: Vec<(String, Value)> = keys
                                .into_iter()
                                .filter(|k| store.has_edge_prop(eid, k))
                                .map(|k| {
                                    let v = store.edge_prop(eid, &k);
                                    (k, v)
                                })
                                .collect();
                            pairs.sort_by(|a, b| a.0.cmp(&b.0));
                            let mut entries: Vec<(Value, Value)> = Vec::new();
                            if tokens {
                                entries.push((
                                    Value::Str(GStr::from("id")),
                                    store.edge_ext_id(eid).map_or(Value::Null, Value::Str),
                                ));
                                entries.push((
                                    Value::Str(GStr::from("label")),
                                    store.edge_type_name(eid).map_or(Value::Null, |t| {
                                        Value::Str(GStr::from(t.as_str()))
                                    }),
                                ));
                            }
                            entries.extend(pairs.into_iter().map(|(k, v)| {
                                let v = if wrap { Value::List(vec![v]) } else { v };
                                (Value::Str(k.into()), v)
                            }));
                            Value::Map(Arc::new(entries))
                        }
                        _ => Value::Null,
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
                // The Gremlin vertex path: each hop's node as its full element map.
                // Read the node-id lineage DIRECTLY — `args[0]` is `Expr::Path` (kept so
                // `needs_lineage` fires), but its GQL value is now a rich Path object.
                let n = batch.rows();
                let out: Vec<Value> = match &batch.lineage {
                    Some(lin) => (0..n)
                        .map(|i| Value::List(path_node_values(store, lin.path_at(i))))
                        .collect(),
                    None => vec![Value::Null; n],
                };
                return Ok(Col::Gen(out));
            }
            // `path_values(path, 'k')` → Gremlin `path().by('k')`: render each path
            // element as its `k` property instead of the whole vertex element map.
            if name == "path_values" {
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
                // Read the node-id lineage directly (see `path_nodes`).
                let out: Vec<Value> = match &batch.lineage {
                    Some(lin) => (0..n)
                        .map(|i| {
                            Value::List(
                                lin.path_at(i)
                                    .iter()
                                    .map(|v| match v {
                                        Value::Num(id) => map_elem(*id as u32),
                                        other => other.clone(),
                                    })
                                    .collect(),
                            )
                        })
                        .collect(),
                    None => vec![Value::Null; n],
                };
                return Ok(Col::Gen(out));
            }
            // `path_has_dup(path)` → Gremlin `cyclicPath`/`simplePath` support: TRUE if
            // the lineage node path repeats any vertex, FALSE if all distinct. The
            // argument is `Expr::Path` (a per-row list of node-id Nums); a Null row
            // (no lineage) is Null. Gremlin-only — not in the GQL whitelist.
            if name == "path_has_dup" {
                // Read the node-id lineage directly (see `path_nodes`): TRUE if a vertex
                // repeats, FALSE if all distinct, NULL when no lineage is tracked.
                let n = batch.rows();
                let out: Vec<Value> = match &batch.lineage {
                    Some(lin) => (0..n)
                        .map(|i| {
                            let mut seen: std::collections::HashSet<u64> =
                                std::collections::HashSet::new();
                            let dup = lin.path_at(i).iter().any(|v| match v {
                                Value::Num(id) => !seen.insert(id.to_bits()),
                                _ => false,
                            });
                            Value::Bool(dup)
                        })
                        .collect(),
                    None => vec![Value::Null; n],
                };
                return Ok(Col::Gen(out));
            }
            // `list_{sum,mean,min,max}(list)` → Gremlin's scope-LOCAL aggregates over
            // a list cell (e.g. after `fold()`): reduce the list's NUMERIC elements
            // (nulls/non-numerics skipped), yielding Null for a list with no number —
            // matching the TS engine's `local_num`/`local_extreme` on the numeric case.
            // Gremlin-only. (Mixed numeric+non-numeric lists are the held cross-type
            // territory; here the non-numerics are simply skipped.)
            // `list_count(list)` → Gremlin `count(local)`: the number of local
            // elements (a list's length, or 1 for a scalar cell — the TS engine's
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
                        // labels — both as string lists (matching the TS engine).
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
                            .map(|((k, _), c)| (GStr::from(k.as_str()), c.value_at(i)))
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
                                (Value::Str(GStr::from(k.as_str())), render_cell(c, i, store))
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
            let otherwise = otherwise.as_deref();
            // Eager fast path over the FULL batch, INLINE so it keeps the plain-CASE codegen
            // (fully vectorized). On ANY error a branch may hold a type error for a row that
            // never actually takes it, so fall back to the lazy (SQL-standard) masked
            // evaluation, which evaluates each condition/value only over the rows that reach
            // it — an unreached branch's type error cannot surface (`CASE WHEN true THEN 1
            // ELSE (2 + 'abc') END` is 1). The masked path runs ONLY on the error.
            let eager: Result<Col, String> = (|| {
                // Categorical remap fast path: every branch is `<dict col> = <str literal>`.
                // Select by code without evaluating a full compare column per branch.
                if let Some((slot, key, code_to_branch)) = case_dict_lookup(branches, store) {
                    if let (Some(Column::Dict { codes, present, .. }), Col::Nodes(ids)) =
                        (store.column(&key), batch.slot(slot))
                    {
                        let vals = eval_all(branches.iter().map(|(_, v)| v), store, batch)?;
                        let else_col = otherwise.map(|e| eval(e, store, batch)).transpose()?;
                        let out: Vec<Value> = ids
                            .iter()
                            .enumerate()
                            .map(|(i, &id)| {
                                let bi = (id != u32::MAX && present[id as usize])
                                    .then(|| code_to_branch[codes[id as usize] as usize])
                                    .flatten();
                                match bi {
                                    Some(b) => vals[b].value_at(i),
                                    None => {
                                        else_col.as_ref().map_or(Value::Null, |c| c.value_at(i))
                                    }
                                }
                            })
                            .collect();
                        return Ok(Col::Gen(out));
                    }
                }
                let conds = eval_all(branches.iter().map(|(c, _)| c), store, batch)?;
                let vals = eval_all(branches.iter().map(|(_, v)| v), store, batch)?;
                let else_col = otherwise.map(|e| eval(e, store, batch)).transpose()?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        // First branch whose condition is TRUE (three-valued). A non-null
                        // non-boolean condition is a data exception (a WHEN must be boolean).
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
                Ok(Col::Gen(out))
            })();
            match eager {
                Ok(col) => col,
                Err(_) => eval_case_masked(branches, otherwise, store, batch)?,
            }
        }
        Expr::Cast { target, expr } => {
            // Evaluate the input, then cast per row via the value contract. A
            // failed conversion aborts the whole evaluation (E_INVALID_VALUE) —
            // the read pipeline is fallible precisely so this can throw.
            let col = eval(expr, store, batch)?;
            let t = (*target).to_value();
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
                Col::Edges(eids) => {
                    // Match ANY of the wanted labels against the edge's WHOLE label set
                    // (primary OR secondary) — a multi-label edge `[KNOWS, CREATED]` is
                    // `hasLabel('CREATED')` too. `edge_type_name` alone saw only the
                    // primary type and missed the rest. Resolve the wanted type ids once.
                    let want: Vec<u32> = labels.iter().filter_map(|l| store.etype_id(l)).collect();
                    Col::Bool(
                        eids.iter()
                            .map(|&e| {
                                e != u32::MAX && want.iter().any(|&t| store.edge_carries_type(e, t))
                            })
                            .collect(),
                    )
                }
                // A heterogeneous Col::Gen (a mixed branch / inject) carries UNBOXED element
                // refs: test the label per Node/Edge cell; a scalar cell is false (has no label),
                // matching the pure-TS heterogeneous stream.
                Col::Gen(cells) => {
                    let want_e: Vec<u32> =
                        labels.iter().filter_map(|l| store.etype_id(l)).collect();
                    Col::Bool(
                        cells
                            .iter()
                            .map(|c| match c {
                                Value::Node(id) => {
                                    node_buckets.iter().any(|b| b.binary_search(id).is_ok())
                                }
                                Value::Edge(e) => {
                                    want_e.iter().any(|&t| store.edge_carries_type(*e, t))
                                }
                                _ => false,
                            })
                            .collect(),
                    )
                }
                other => Col::Bool(vec![false; other.len()]),
            }
        }
        Expr::PropertyExists { slot, key } => {
            // Presence, not value: TRUE iff the element carries a stored value for
            // `key`, FALSE if not — but on a NON-element (the OPTIONAL null sentinel
            // `u32::MAX`, or a computed value) the answer is NULL, matching the TS engine's
            // `prop_present` (`_ => Val::Null`). A column with no sentinel keeps the
            // unboxed `Col::Bool` fast path; a sentinel forces the null-carrying `Gen`.
            // A slot past the runtime width (a branch/inject collapsed the layout) has no
            // element to test — the presence is NULL, like a non-element frontier.
            if *slot >= batch.slots.len() {
                return Ok(Col::Gen(vec![Value::Null; batch.rows()]));
            }
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
                // A heterogeneous column (e.g. post-union `Gen` of boxed element maps):
                // presence reads through a boxed vertex/edge's `properties`; a genuine
                // non-element value has no property and stays NULL (the TS engine's `_ => Null`).
                other => {
                    Col::Gen(
                        (0..other.len())
                            .map(|i| match other.value_at(i) {
                                // An UNBOXED element ref reads presence off the store.
                                Value::Node(id) => Value::Bool(store.has_prop(id, key)),
                                Value::Edge(e) => Value::Bool(store.has_edge_prop(e, key)),
                                Value::Map(pairs) => match boxed_element_props(&pairs) {
                                    Some(props) => Value::Bool(props.iter().any(
                                        |(k, _)| matches!(k, Value::Str(s) if s.as_ref() == key),
                                    )),
                                    None => Value::Null,
                                },
                                _ => Value::Null,
                            })
                            .collect(),
                    )
                }
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
            if let Some(Col::Num(ids)) = survivors.slots.get(prov) {
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
            if let Some(Col::Num(ids)) = survivors.slots.get(prov) {
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
            if let Some(Col::Num(ids)) = survivors.slots.get(prov).cloned() {
                for (j, &id) in ids.iter().enumerate() {
                    let i = id as usize;
                    if i < n {
                        out[i].push(render_cell(&vals, j, store));
                    }
                }
            }
            Col::Gen(out.into_iter().map(Value::List).collect())
        }
        Expr::AggSubquery {
            body, scalar, func, ..
        } => {
            // Correlated scalar aggregate: the same provenance-tagged sub-run as
            // CollectSubquery, but REDUCE `scalar` per outer row under `func` instead of
            // gathering into a list. Empty / all-null group → NULL (SQL aggregate-of-
            // nothing) for sum/avg/min/max.
            let n = batch.rows();
            let prov = batch.slots.len();
            let mut slots = batch.slots.clone();
            slots.push(Col::Num((0..n).map(|i| i as f64).collect()));
            let seed = Batch::of(slots);
            let survivors = pull_body(body, store, &seed)?;
            let vals = eval(scalar, store, &survivors)?;
            // Per outer row: (running total, count of numeric values, best for min/max).
            let mut acc: Vec<(f64, u64, Option<f64>)> = vec![(0.0, 0, None); n];
            if let Some(Col::Num(ids)) = survivors.slots.get(prov).cloned() {
                for (j, &id) in ids.iter().enumerate() {
                    let i = id as usize;
                    if i >= n {
                        continue;
                    }
                    let Value::Num(x) = vals.value_at(j) else {
                        continue; // NULL / non-numeric args are ignored (SQL sum/avg/min/max)
                    };
                    let (total, cnt, best) = &mut acc[i];
                    *total += x;
                    *cnt += 1;
                    *best = Some(match *best {
                        None => x,
                        Some(b) => {
                            let take_new = (*func == AggFn::Min
                                && value::cmp_num_total(x, b).is_lt())
                                || (*func == AggFn::Max && value::cmp_num_total(x, b).is_gt());
                            if take_new {
                                x
                            } else {
                                b
                            }
                        }
                    });
                }
            }
            let out: Vec<Value> = acc
                .into_iter()
                .map(|(total, cnt, best)| match func {
                    // GQL `SUM` of an empty / all-null set is 0 (not SQL's NULL) — matching
                    // pure-TS and the engine's ordinary aggregate.
                    AggFn::Sum => Value::Num(total),
                    // AVG / MIN / MAX of nothing → NULL.
                    AggFn::Avg if cnt == 0 => Value::Null,
                    AggFn::Avg => Value::Num(total / cnt as f64),
                    _ => best.map_or(Value::Null, Value::Num),
                })
                .collect();
            Col::Gen(out)
        }
        Expr::ScalarSubquery { body, scalar, .. } => {
            // Correlated scalar: same provenance-tagged sub-run, but project `scalar`
            // over the surviving sub-rows and return each outer row's single value
            // (NULL when the body matched nothing). A VALUE subquery must return AT
            // MOST one row per outer row — more than one is an error (matching the TS engine).
            let n = batch.rows();
            let prov = batch.slots.len();
            let mut slots = batch.slots.clone();
            slots.push(Col::Num((0..n).map(|i| i as f64).collect()));
            let seed = Batch::of(slots);
            let survivors = pull_body(body, store, &seed)?;
            let vals = eval(scalar, store, &survivors)?;
            let mut out = vec![Value::Null; n];
            let mut seen = vec![false; n];
            if let Some(Col::Num(ids)) = survivors.slots.get(prov).cloned() {
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
