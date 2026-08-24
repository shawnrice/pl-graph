use super::*;
use crate::batch::{Batch, Col};
use crate::store::Store;
use crate::value::Value;

pub(super) fn push_group_cols(
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
pub(super) struct StepRec {
    levels: Vec<(u32, usize)>,
    source: u32,
    edge: u32,
    target: u32,
}

/// A partially-built nested list keyed by a rep-tuple: `insert([i,j], v)` puts `v` at
/// `list[i][j]`, growing intermediate lists. Depth-`d` variable → `d+1`-element keys.
pub(super) enum Nest {
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
pub(super) fn bind_nested(
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
pub(super) fn nested_group(
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
