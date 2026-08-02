//! Pattern traversal: variable-length walks, shortest-path, and bidirectional
//! reachability. Extracted from the evaluator (`super`) as one cohesive chapter;
//! shares its execution context (`Ctx`/`Env`/`Val`/`Binding`) and helpers via
//! `use super::*` (a child module sees its parent's private items).
use super::*;

/// Every *trail* endpoint — a path traversing each relationship at most once
/// (ISO/IEC 39075 default for a quantified path) — from `from` within [min, max]
/// hops of `rel`, streamed to `on_end` in trail-discovery order. An endpoint
/// reached by `k` distinct trails is emitted `k` times (ISO per-path
/// multiplicity); `min == 0` emits the zero-length trail (the start node) first.
///
/// **Lazy / short-circuiting:** `on_end` returns `false` to stop the walk; this
/// returns `false` when it did (propagating a consumer's stop, e.g. `EXISTS`
/// found a witness or `LIMIT` filled), `true` when the trails were exhausted.
/// Streaming (not collecting into a `Vec`) is what lets `EXISTS`/`LIMIT` avoid
/// enumerating an exponential trail set on a dense graph — the eager version hit
/// the trail budget and faulted where a single witness sufficed.
///
/// Iterative (explicit stack) so a long chain can't overflow the native stack;
/// edge-uniqueness bounds trail length to the edge count, so it always terminates
/// on cycles. The number of trails can still be exponential, so when a consumer
/// does need them all a per-expansion step budget records `FAULT_BUDGET`
/// (→ `ResourceExhausted`) and stops rather than exhausting memory/time.
/// The traversal shape for `reachable_each`: how far to repeat (`q`), which
/// repeats are legal (`mode`), and whether the consumer needs each trail's full
/// `(vertices, edges)` reconstructed (`want_path`) or just its endpoint.
#[derive(Clone, Copy)]
pub(super) struct WalkSpec {
    pub(super) q: Quantifier,
    pub(super) mode: PathMode,
    // When `true`, reconstruct each trail's `(vertices, edges)` from the frame
    // stack and pass them to `on_end` (a path variable needs the whole walk);
    // otherwise pass empty slices (endpoint-only consumers skip the O(depth)
    // rebuild). The two are byte-identical — same enumeration, one just carries
    // the path.
    pub(super) want_path: bool,
}

/// `reachable_each`'s per-trail sink: `(binding, endpoint, vertices, edges, steps) ->
/// keep-going`. The binding is threaded through so the driver can bind each hop's
/// edge for a per-hop predicate; the sink binds the endpoint node on top of it.
/// `vertices`/`edges` are the FLAT walk (for a Path value); `steps` is the same walk
/// tagged with each hop's position in the (possibly nested) repetition pattern, so a
/// subpath sink can assemble group variables at any nesting depth (see [`StepRec`]).
type OnEnd<'a> = &'a mut dyn FnMut(&mut Binding, u32, &[u32], &[u32], &[StepRec]) -> bool;

/// Does edge `eidx` satisfy the segment's per-hop predicate (inline properties +
/// `WHERE`)? The optional edge variable is bound to this edge for the duration of
/// the check so the predicate can name it (`e.amt > $t`); outer bound variables in
/// `binding` stay visible, and the slot is restored afterward. (For a parenthesized
/// subpath the per-repetition source/target binding is done by the unit walk in
/// [`expand_unit`], not here.)
pub(super) fn edge_passes(
    graph: &Graph,
    ctx: &Ctx,
    binding: &mut Binding,
    rel: &CRel,
    eidx: u32,
) -> bool {
    if rel.props.is_empty() && rel.where_.is_none() {
        return true;
    }
    let restore = rel.var_slot.map(|s| {
        let prev = binding.get(s).cloned();
        binding.set(s, Val::Edge(eidx));
        (s, prev)
    });
    let ok = satisfies(
        graph,
        ctx,
        &Val::Edge(eidx),
        &rel.props,
        rel.where_.as_ref(),
        binding,
    );
    if let Some((s, prev)) = restore {
        match prev {
            Some(v) => binding.set(s, v),
            None => binding.unset(s),
        }
    }
    ok
}

/// Expand one segment from `v`, keeping only edges that pass the per-hop predicate
/// (when the segment carries one). Materialised into a `Vec` because the DFS stack
/// needs a resumable per-frame cursor.
pub(super) fn expand_filtered(
    graph: &Graph,
    ctx: &Ctx,
    binding: &mut Binding,
    rel: &CRel,
    v: u32,
) -> Vec<(u32, u32)> {
    if rel.props.is_empty() && rel.where_.is_none() {
        return expand(graph, ctx, v, rel.direction, rel.label.as_ref()).collect();
    }
    expand(graph, ctx, v, rel.direction, rel.label.as_ref())
        .filter(|(eidx, _)| edge_passes(graph, ctx, binding, rel, *eidx))
        .collect()
}

/// One graph-consuming hop of a matched trail, tagged with its position in the
/// (possibly nested) repetition pattern. `levels` is the cursor stack outer→inner:
/// one `(rep, elem_after)` per active unit, where `elem_after` is the element index
/// the hop ADVANCED PAST (so the hop's own element is `elem_after - 1`, and a step
/// sitting inside a `Sub` keeps the outer unit's `elem` pinned at that Sub's index).
/// This is what lets ONE structured binder assemble group variables at any nesting
/// depth, instead of striding a flat array by a fixed `k`.
pub(super) struct StepRec {
    levels: Vec<(u32, usize)>,
    source: u32,
    edge: u32,
    target: u32,
}

/// A partially-built nested list, indexed by a rep-tuple. `insert([i,j], v)` places
/// `v` at `list[i][j]`, growing intermediate lists as needed. A depth-0 variable has
/// 1-element index tuples (a flat list); a variable nested `d` quantifiers deep has
/// `d+1`-element tuples (a list nested `d+1` deep).
pub(super) enum Nest {
    Leaf(Val),
    List(Vec<Self>),
}

impl Nest {
    fn empty() -> Self {
        Self::List(Vec::new())
    }
    fn insert(&mut self, idx: &[u32], val: Val) {
        match idx.split_first() {
            None => *self = Self::Leaf(val),
            Some((&i, rest)) => {
                if !matches!(self, Self::List(_)) {
                    *self = Self::List(Vec::new());
                }
                if let Self::List(v) = self {
                    let i = i as usize;
                    while v.len() <= i {
                        v.push(Self::List(Vec::new()));
                    }
                    v[i].insert(rest, val);
                }
            }
        }
    }
    fn into_val(self) -> Val {
        match self {
            Self::Leaf(v) => v,
            Self::List(items) => Val::List(items.into_iter().map(Self::into_val).collect()),
        }
    }
}

/// The HOT-path group-variable binder for a FLAT (all-`Hop`) unit: the walk is `r`
/// repetitions of a fixed `k`-hop unit, so `verts` = `[seed, …]` of length `r·k + 1` and
/// `edges` of length `r·k`; the variable at node position `p` (0 = source `x`, … `k` = last
/// target) is `verts[rep·k + p]` across reps, at edge position `p` is `edges[rep·k + p]`.
/// A direct stride over two flat arrays — no per-hop `StepRec`/`levels`/`Nest` allocation
/// (that generality is only needed for nesting; see [`bind_group_vars`]). Byte-identical to
/// the structured binder on a flat unit (conformance-guarded).
pub(super) fn bind_group_vars_flat(
    binding: &mut Binding,
    unit: &CUnit,
    verts: &[u32],
    edges: &[u32],
) -> Vec<(usize, Option<Val>)> {
    let k = unit.elems.len();
    let reps = edges.len().checked_div(k).unwrap_or(0);
    let mut restores = Vec::new();
    let mut bind = |binding: &mut Binding, slot: Option<usize>, list: Vec<Val>| {
        if let Some(s) = slot {
            restores.push((s, binding.get(s).cloned()));
            binding.set(s, Val::List(list));
        }
    };
    for p in 0..=k {
        let slot = if p == 0 {
            unit.start_slot
        } else {
            unit.hop(p - 1).target_slot
        };
        bind(
            binding,
            slot,
            (0..reps).map(|rep| Val::Node(verts[rep * k + p])).collect(),
        );
    }
    for p in 0..k {
        bind(
            binding,
            unit.hop(p).rel.var_slot,
            (0..reps).map(|rep| Val::Edge(edges[rep * k + p])).collect(),
        );
    }
    restores
}

/// Expose a quantified subpath's inner variables as GROUP variables from the
/// structured walk (the GENERAL path, for NESTED units). Each variable becomes the
/// (possibly nested) list of its value over every repetition — one list level per
/// enclosing quantifier (a `Sub`'s inner variables nest one level deeper; a `Sub`'s
/// landing is its last inner hop's target). A flat unit takes [`bind_group_vars_flat`]
/// instead (same result, no per-hop allocation). Returns prior slot values to restore.
pub(super) fn bind_group_vars(
    binding: &mut Binding,
    unit: &CUnit,
    steps: &[StepRec],
) -> Vec<(usize, Option<Val>)> {
    let mut restores = Vec::new();
    bind_unit(binding, unit, &[], 0, steps, &mut restores);
    restores
}

/// Bind ONE outer rep's variables for the per-repetition `WHERE` (`steps` already
/// filtered to that rep). `key_start = 1` drops the outer-rep index from every key, so a
/// variable at the OUTER unit's depth collapses to a SCALAR (this rep's single value) and
/// a variable inside a `Sub` becomes a LIST over the inner reps — exactly the per-rep view
/// the predicate sees (`size(e)`, `x[0]`, …).
pub(super) fn bind_group_vars_perrep(
    binding: &mut Binding,
    unit: &CUnit,
    steps: &[StepRec],
) -> Vec<(usize, Option<Val>)> {
    let mut restores = Vec::new();
    bind_unit(binding, unit, &[], 1, steps, &mut restores);
    restores
}

/// Assemble one unit's group variables. `tree_path` is the sequence of `Sub`-element
/// indices from the top unit down to THIS unit, so `depth = tree_path.len()` is the
/// unit's nesting depth (0 = top). A variable here is indexed by the rep counters of
/// levels `key_start..=depth`: the enclosing quantifiers form the outer list dimensions,
/// this unit's own rep is the innermost. `key_start = 0` is the full-nesting emit binding;
/// `key_start = 1` drops the outer level for a per-rep `WHERE` view. `steps` is in walk
/// order, so "first/last matching step" needs no sort. Recurses into each `Sub` child.
pub(super) fn bind_unit(
    binding: &mut Binding,
    unit: &CUnit,
    tree_path: &[usize],
    key_start: usize,
    steps: &[StepRec],
    restores: &mut Vec<(usize, Option<Val>)>,
) {
    let depth = tree_path.len();
    // The rep-tuple that indexes a variable of THIS unit: reps of levels `key_start..=depth`.
    let key = |s: &StepRec| -> Vec<u32> {
        s.levels[key_start..depth + 1]
            .iter()
            .map(|(r, _)| *r)
            .collect()
    };
    // Is `s` a hop somewhere inside this unit (at this tree position), possibly deeper?
    let within = |s: &StepRec| -> bool {
        s.levels.len() > depth
            && s.levels[..depth]
                .iter()
                .map(|(_, e)| *e)
                .eq(tree_path.iter().copied())
    };

    // `start_slot` = the source vertex of each rep-instance = its FIRST hop's source
    // (which may sit inside a leading `Sub`, hence `within`, not just direct hops).
    if let Some(slot) = unit.start_slot {
        let mut nest = Nest::empty();
        let mut seen: FxHashSet<Vec<u32>> = FxHashSet::default();
        for s in steps.iter().filter(|s| within(s)) {
            let k = key(s);
            if seen.insert(k.clone()) {
                nest.insert(&k, Val::Node(s.source));
            }
        }
        restores.push((slot, binding.get(slot).cloned()));
        binding.set(slot, nest.into_val());
    }

    for (e, elem) in unit.elems.iter().enumerate() {
        match elem {
            CElem::Hop(h) => {
                // A DIRECT hop at element `e`: one level deeper than the enclosing
                // path, its own `elem_after` == e + 1.
                let direct = |s: &&StepRec| {
                    within(s) && s.levels.len() == depth + 1 && s.levels[depth].1 == e + 1
                };
                if let Some(slot) = h.target_slot {
                    let mut nest = Nest::empty();
                    for s in steps.iter().filter(direct) {
                        nest.insert(&key(s), Val::Node(s.target));
                    }
                    restores.push((slot, binding.get(slot).cloned()));
                    binding.set(slot, nest.into_val());
                }
                if let Some(slot) = h.rel.var_slot {
                    let mut nest = Nest::empty();
                    for s in steps.iter().filter(direct) {
                        nest.insert(&key(s), Val::Edge(s.edge));
                    }
                    restores.push((slot, binding.get(slot).cloned()));
                    binding.set(slot, nest.into_val());
                }
            }
            CElem::Sub(sub) => {
                // A `Sub`'s landing = the target of its LAST inner hop, per rep-instance
                // (inner steps keep this unit's `elem` pinned at `e`).
                if let Some(slot) = sub.target_slot {
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
                    let mut nest = Nest::empty();
                    for (k, t) in last {
                        nest.insert(&k, Val::Node(t));
                    }
                    restores.push((slot, binding.get(slot).cloned()));
                    binding.set(slot, nest.into_val());
                }
                let mut child_path = tree_path.to_vec();
                child_path.push(e);
                bind_unit(binding, &sub.unit, &child_path, key_start, steps, restores);
            }
        }
    }
}

/// The GENERAL repetition matcher: repeat `unit` from `from`, streaming each trail end
/// in `[min, max]` REPETITIONS to `on_end` (endpoint + the whole walk's `verts`/`edges`
/// for group-variable / path exposure). `unit`'s elements are hops OR nested quantified
/// sub-units, so this is ONE matcher for every var-length shape — abbreviated,
/// multi-element, and nested `( … ){n,m}`.
///
/// It is a single, LAZY, explicit-stack pushdown DFS. The stack holds one frame per hop
/// of the CURRENT path only — never the `d^k` traversals from a vertex — so memory is
/// O(path length) and there is no native recursion (no stack overflow on a deep walk).
/// A frame's pattern **position** is a stack of loop cursors (`Position`), and `resolve`
/// follows the epsilon moves (enter a sub / close a nested unit / repeat) to the graph-
/// consuming hops reachable from it; a visited set breaks min-0 epsilon cycles.
/// TRAIL/SIMPLE/ACYCLIC/WALK restrictors are applied PER HOP.
///
/// Performance: the linear world (a depth-1 `Position::Flat`, no branching) is the hot
/// case, so `resolve`'s fast path computes it inline with no allocation, the position is
/// borrowed not copied per hop, and moves are a `One`/`Many` enum (no vector for the
/// common single move). Subpath / multi-element walks run at the old fused matcher's
/// speed; only the tightest abbreviated loop pays a small (~10%) per-hop overhead for
/// carrying the general position — the price of one code path instead of a special case.
pub(super) fn reachable_each_unit(
    graph: &Graph,
    ctx: &Ctx,
    binding: &mut Binding,
    from: u32,
    unit: &CUnit,
    spec: WalkSpec,
    on_end: OnEnd<'_>,
) -> bool {
    let WalkSpec { q, mode, want_path } = spec;
    if ctx.faulted() {
        return true;
    }
    if q.min == 0 && !on_end(binding, from, &[from], &[], &[]) {
        return false;
    }

    let trail = matches!(mode, PathMode::Trail);
    let vertex_mode = matches!(mode, PathMode::Simple | PathMode::Acyclic);
    let mut marks = if trail {
        ctx.take_marks(graph.edge_slots())
    } else if vertex_mode {
        vec![false; graph.vertex_count()]
    } else {
        Vec::new()
    };
    if vertex_mode {
        marks[from as usize] = true;
    }

    // A loop cursor: which unit, its `[min,max]`, completed reps, next element.
    #[derive(Clone, Copy)]
    struct Cursor<'a> {
        unit: &'a CUnit,
        min: u32,
        // `u32::MAX` = unbounded (`*` / `+` / `{n,}`) — a sentinel, so the cursor stays
        // small and `Copy` (no `Option` on the hot path).
        max: u32,
        rep: u32,
        elem: usize,
    }

    // A position in the (possibly nested) pattern — a cursor stack, innermost last.
    // `Flat` is the depth-1 case (the entire linear world): a single `Copy` cursor with
    // NO heap allocation. Nesting deepens to `Deep`.
    #[derive(Clone)]
    enum Position<'a> {
        Flat(Cursor<'a>),
        Deep(Vec<Cursor<'a>>),
    }

    impl<'a> Position<'a> {
        fn top(&self) -> &Cursor<'a> {
            match self {
                Position::Flat(c) => c,
                Position::Deep(v) => v.last().expect("a position has ≥ 1 cursor"),
            }
        }
        fn top_mut(&mut self) -> &mut Cursor<'a> {
            match self {
                Position::Flat(c) => c,
                Position::Deep(v) => v.last_mut().expect("a position has ≥ 1 cursor"),
            }
        }
        fn len(&self) -> usize {
            match self {
                Position::Flat(_) => 1,
                Position::Deep(v) => v.len(),
            }
        }
        fn push(&mut self, c: Cursor<'a>) {
            match self {
                Position::Flat(old) => *self = Position::Deep(vec![*old, c]),
                Position::Deep(v) => v.push(c),
            }
        }
        fn pop(&mut self) {
            if let Position::Deep(v) = self {
                v.pop();
                if v.len() == 1 {
                    *self = Position::Flat(v[0]);
                }
            }
        }
        fn key(&self) -> Vec<(usize, u32, usize)> {
            let one = |c: &Cursor| (c.unit as *const CUnit as usize, c.rep, c.elem);
            match self {
                Position::Flat(c) => vec![one(c)],
                Position::Deep(v) => v.iter().map(one).collect(),
            }
        }
        // The `(rep, elem)` per active unit, outer→inner — the tag [`StepRec`] carries
        // so one structured binder can place a hop's value at its nesting depth.
        fn levels(&self) -> Vec<(u32, usize)> {
            let one = |c: &Cursor| (c.rep, c.elem);
            match self {
                Position::Flat(c) => vec![one(c)],
                Position::Deep(v) => v.iter().map(one).collect(),
            }
        }
        // The OUTERMOST unit's rep index — which outer repetition this position is in.
        fn outer_rep(&self) -> u32 {
            match self {
                Position::Flat(c) => c.rep,
                Position::Deep(v) => v[0].rep,
            }
        }
    }

    struct HopMove<'a> {
        rel: &'a CRel,
        after: Position<'a>,
    }

    // The graph-consuming moves from a position. `One` (the linear case) never allocates
    // a vector; `Many` is the general branching case (nesting).
    enum MoveSet<'a> {
        None,
        One(HopMove<'a>),
        Many(Vec<HopMove<'a>>),
    }

    impl<'a> MoveSet<'a> {
        fn get(&self, i: usize) -> Option<&HopMove<'a>> {
            match self {
                MoveSet::None => None,
                MoveSet::One(m) => (i == 0).then_some(m),
                MoveSet::Many(v) => v.get(i),
            }
        }
        fn is_empty(&self) -> bool {
            matches!(self, MoveSet::None)
        }
        // Drop moves that ADVANCE past outer rep `rep` (they start the next outer rep) —
        // used when a failed per-rep `WHERE` prunes the outer-completion branch but the
        // inner-continue branches (outer rep == `rep`) must survive.
        fn retain_outer_le(self, rep: u32) -> Self {
            match self {
                MoveSet::None => MoveSet::None,
                MoveSet::One(m) => {
                    if m.after.outer_rep() <= rep {
                        MoveSet::One(m)
                    } else {
                        MoveSet::None
                    }
                }
                MoveSet::Many(v) => {
                    let kept: Vec<_> = v
                        .into_iter()
                        .filter(|m| m.after.outer_rep() <= rep)
                        .collect();
                    if kept.is_empty() {
                        MoveSet::None
                    } else {
                        MoveSet::Many(kept)
                    }
                }
            }
        }
    }

    // Follow epsilon moves (enter a sub / close a nested unit / repeat) from `start`,
    // collecting whether the TOP unit accepts here (emit) and every graph-consuming hop
    // reachable. A visited set breaks epsilon cycles (a zero-edge min-0 inner loop).
    //
    // A `Flat` (depth-1) position has no epsilon branching — the ENTIRE linear world —
    // so its ≤1 hop move is computed inline here (no worklist / visited-set / heap
    // position); only nesting reaches the out-of-line general resolver.
    // Returns `(emit, completed_outer, moves)`: whether the top unit ACCEPTS here, whether
    // an OUTER rep just completed (the top unit reached its end — the per-rep `WHERE` hook,
    // true even below `min`), and every graph-consuming hop reachable.
    #[inline]
    fn resolve<'a>(start: &Position<'a>) -> (bool, bool, MoveSet<'a>) {
        if let Position::Flat(c) = start {
            let c = *c;
            if c.elem < c.unit.elems.len() {
                if let CElem::Hop(h) = &c.unit.elems[c.elem] {
                    let mut a = c;
                    a.elem += 1;
                    return (
                        false,
                        false,
                        MoveSet::One(HopMove {
                            rel: &h.rel,
                            after: Position::Flat(a),
                        }),
                    );
                }
            } else {
                // The flat unit reached its end — outer rep `c.rep` completed.
                let rep2 = c.rep + 1;
                let emit = rep2 >= c.min;
                if rep2 < c.max {
                    if let CElem::Hop(h) = &c.unit.elems[0] {
                        let mut a = c;
                        a.rep = rep2;
                        a.elem = 1;
                        return (
                            emit,
                            true,
                            MoveSet::One(HopMove {
                                rel: &h.rel,
                                after: Position::Flat(a),
                            }),
                        );
                    }
                } else {
                    return (emit, true, MoveSet::None);
                }
            }
        }
        resolve_general(start)
    }

    // The general epsilon-closure (nesting): kept out of line so the linear fast path
    // above stays small enough to inline at its call sites.
    #[inline(never)]
    fn resolve_general<'a>(start: &Position<'a>) -> (bool, bool, MoveSet<'a>) {
        let mut emit = false;
        let mut completed = false;
        let mut hops: Vec<HopMove<'a>> = Vec::new();
        let mut work = vec![start.clone()];
        let mut seen: Vec<Vec<(usize, u32, usize)>> = Vec::new();
        while let Some(p) = work.pop() {
            let key = p.key();
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let top = *p.top();
            if top.elem < top.unit.elems.len() {
                match &top.unit.elems[top.elem] {
                    CElem::Hop(h) => {
                        let mut after = p.clone();
                        after.top_mut().elem += 1;
                        hops.push(HopMove { rel: &h.rel, after });
                    }
                    CElem::Sub(s) => {
                        let mut enter = p.clone();
                        enter.push(Cursor {
                            unit: &s.unit,
                            min: s.min,
                            max: s.max.unwrap_or(u32::MAX),
                            rep: 0,
                            elem: 0,
                        });
                        work.push(enter);
                        if s.min == 0 {
                            let mut skip = p.clone();
                            skip.top_mut().elem += 1;
                            work.push(skip);
                        }
                    }
                }
            } else {
                let rep2 = top.rep + 1;
                if rep2 >= top.min {
                    if p.len() == 1 {
                        emit = true;
                    } else {
                        let mut close = p.clone();
                        close.pop();
                        close.top_mut().elem += 1;
                        work.push(close);
                    }
                }
                // The TOP unit at its end (any `rep`, `min` or not) = an outer rep completed.
                if p.len() == 1 {
                    completed = true;
                }
                if rep2 < top.max {
                    let mut again = p.clone();
                    let t = again.top_mut();
                    t.rep = rep2;
                    t.elem = 0;
                    work.push(again);
                }
            }
        }
        let moves = if hops.is_empty() {
            MoveSet::None
        } else {
            MoveSet::Many(hops)
        };
        (emit, completed, moves)
    }

    struct Frame<'a> {
        vertex: u32,
        moves: MoveSet<'a>,
        move_idx: usize,
        edges: Vec<(u32, u32)>,
        edge_idx: usize,
        entry_edge: Option<u32>,
        entry_mark: Option<usize>,
    }

    fn clear_live(stack: &[Frame], marks: &mut [bool]) {
        for f in stack {
            if let Some(mi) = f.entry_mark {
                marks[mi] = false;
            }
        }
    }

    fn rebuild(stack: &[Frame], v: u32, e: u32) -> (Vec<u32>, Vec<u32>) {
        let mut pv: Vec<u32> = stack.iter().map(|f| f.vertex).collect();
        pv.push(v);
        let mut pe: Vec<u32> = stack.iter().filter_map(|f| f.entry_edge).collect();
        pe.push(e);
        (pv, pe)
    }

    // The structured walk: one `StepRec` per hop, tagged with its pattern position. Each
    // frame's taken move stays frozen at the hop that spawned its child while that child
    // is live, so `stack[i].moves[move_idx].after` is exactly hop `i`'s landing position
    // (whose top `elem` is one past the element traversed). The final hop (this emit) is
    // `stack[last]`'s current move → `(eidx, nbr)`.
    fn rebuild_steps(stack: &[Frame], nbr: u32, eidx: u32) -> Vec<StepRec> {
        let mut steps = Vec::with_capacity(stack.len());
        for i in 0..stack.len() {
            let at = &stack[i]
                .moves
                .get(stack[i].move_idx)
                .expect("a live move")
                .after;
            let source = stack[i].vertex;
            let (edge, target) = if i + 1 < stack.len() {
                (
                    stack[i + 1].entry_edge.expect("an inner hop has an edge"),
                    stack[i + 1].vertex,
                )
            } else {
                (eidx, nbr)
            };
            steps.push(StepRec {
                levels: at.levels(),
                source,
                edge,
                target,
            });
        }
        steps
    }

    // The top unit's per-repetition `WHERE`, at each OUTER-rep completion. Reconstruct the
    // just-completed rep `rep`'s hops (all steps whose outer level is `rep`), bind the
    // unit's variables to their per-rep values — a direct variable is a SCALAR, a variable
    // inside a nested `Sub` is a LIST over the inner reps (`bind_group_vars_perrep`) — then
    // evaluate. Generalizes the old linear/scalar reconstruction to any nesting.
    let where_ok = |graph: &Graph,
                    ctx: &Ctx,
                    b: &mut Binding,
                    stack: &[Frame],
                    eidx: u32,
                    nbr: u32,
                    rep: u32|
     -> bool {
        let Some(w) = &unit.where_ else {
            return true;
        };
        let rep_steps: Vec<StepRec> = rebuild_steps(stack, nbr, eidx)
            .into_iter()
            .filter(|s| s.levels.first().is_some_and(|(r, _)| *r == rep))
            .collect();
        let restores = bind_group_vars_perrep(b, unit, &rep_steps);
        let ok = as_truth(&eval(&Env::new(graph, ctx, b), w)) == Some(true);
        for (s, prev) in restores.into_iter().rev() {
            match prev {
                Some(v) => b.set(s, v),
                None => b.unset(s),
            }
        }
        ok
    };

    let mut steps: u64 = 0;
    let mut cont = true;
    let has_where = unit.where_.is_some();
    // A flat (all-`Hop`) unit binds group vars by the cheap `k`-stride over the flat walk,
    // so it never needs per-hop `StepRec`s — only a nested unit does.
    let unit_flat = unit.is_flat();

    // (The top unit's `min == 0` zero-rep acceptance is emitted upfront, before marks.)
    let (_, _, seed_moves) = resolve(&Position::Flat(Cursor {
        unit,
        min: q.min,
        max: q.max.unwrap_or(u32::MAX),
        rep: 0,
        elem: 0,
    }));
    let seed_edges = match seed_moves.get(0) {
        Some(m) => expand_filtered(graph, ctx, binding, m.rel, from),
        None => Vec::new(),
    };
    let mut stack: Vec<Frame> = vec![Frame {
        vertex: from,
        moves: seed_moves,
        move_idx: 0,
        edges: seed_edges,
        edge_idx: 0,
        entry_edge: None,
        entry_mark: None,
    }];

    while !stack.is_empty() {
        let li = stack.len() - 1;
        if stack[li].edge_idx >= stack[li].edges.len() {
            // Move to the next resolved hop-move, or backtrack.
            let next = stack[li].move_idx + 1;
            if let Some(m) = stack[li].moves.get(next) {
                let v = stack[li].vertex;
                let e = expand_filtered(graph, ctx, binding, m.rel, v);
                stack[li].move_idx = next;
                stack[li].edges = e;
                stack[li].edge_idx = 0;
                continue;
            }
            if let Some(mi) = stack[li].entry_mark {
                marks[mi] = false;
            }
            stack.pop();
            continue;
        }

        let (eidx, nbr) = stack[li].edges[stack[li].edge_idx];
        stack[li].edge_idx += 1;
        // Borrow the current move's landing position — no per-hop copy (only the general
        // resolver clones it, and only under nesting).
        let after: &Position = &stack[li]
            .moves
            .get(stack[li].move_idx)
            .expect("an in-range move")
            .after;

        // Does this hop finish the TOP unit's rep? (Linear: the position is a single
        // cursor sitting at its unit's end.)
        let completes_top = {
            let t = after.top();
            after.len() == 1 && t.elem == t.unit.elems.len()
        };
        let is_close = matches!(mode, PathMode::Simple) && completes_top && nbr == from;

        if !is_close {
            let collide = match mode {
                PathMode::Trail => marks[eidx as usize],
                PathMode::Simple | PathMode::Acyclic => marks[nbr as usize],
                PathMode::Walk => false,
            };
            if collide {
                continue;
            }
        }

        // Resolve the epsilon-closure: does the top unit ACCEPT here, did an OUTER rep just
        // complete (the per-rep `WHERE` hook), and the onward hops.
        let outer_rep = after.outer_rep();
        let (mut emit, completed_outer, mut next_moves) = resolve(after);

        // Per-repetition `WHERE` at each outer-rep completion. On failure, PRUNE only the
        // outer-completion branch: suppress the emit and drop the moves that start the next
        // outer rep, while inner-continue branches (same outer rep, their rep not yet done)
        // survive. For a linear unit there is no inner branch, so this prunes the whole hop
        // (equivalent to the old per-rep skip).
        if completed_outer
            && has_where
            && !where_ok(graph, ctx, binding, &stack, eidx, nbr, outer_rep)
        {
            emit = false;
            next_moves = next_moves.retain_outer_le(outer_rep);
        }

        steps += 1;
        if steps > graph.limits().trail {
            ctx.set_fault(FAULT_BUDGET);
            clear_live(&stack, &mut marks);
            break;
        }

        let mark = match (is_close, mode) {
            (true, _) | (_, PathMode::Walk) => None,
            (_, PathMode::Trail) => Some(eidx as usize),
            (_, PathMode::Simple | PathMode::Acyclic) => Some(nbr as usize),
        };
        if let Some(mi) = mark {
            marks[mi] = true;
        }

        if emit {
            let (pv, pe, steps) = if want_path {
                let (pv, pe) = rebuild(&stack, nbr, eidx);
                // Only a NESTED unit needs the structured per-hop steps; a flat unit binds
                // from `pv`/`pe` directly (the hot path — no per-hop allocation).
                let steps = if unit_flat {
                    Vec::new()
                } else {
                    rebuild_steps(&stack, nbr, eidx)
                };
                (pv, pe, steps)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
            if !on_end(binding, nbr, &pv, &pe, &steps) {
                cont = false;
                if let Some(mi) = mark {
                    marks[mi] = false;
                }
                clear_live(&stack, &mut marks);
                break;
            }
        }

        // A SIMPLE close emits but does NOT extend (the cycle is closed); likewise a
        // position with no onward move. Nothing to descend into.
        if is_close || next_moves.is_empty() {
            if let Some(mi) = mark {
                marks[mi] = false;
            }
            continue;
        }

        let edges = expand_filtered(
            graph,
            ctx,
            binding,
            next_moves.get(0).expect("non-empty move set").rel,
            nbr,
        );
        stack.push(Frame {
            vertex: nbr,
            moves: next_moves,
            move_idx: 0,
            edges,
            edge_idx: 0,
            entry_edge: Some(eidx),
            entry_mark: mark,
        });
    }

    if trail {
        ctx.return_marks(marks);
    }
    cont
}

/// A single edge is a **one-hop repetition unit**, so the abbreviated `-[]->{n,m}`
/// form is just [`reachable_each_unit`] with a `k = 1` unit built on the fly — there is
/// ONE traversal implementation, no hand-tuned twin to drift. (The one-hop unit is also
/// FASTER than the old bespoke fast-path was: ~102µs vs ~177µs on `bench_k1_abbreviated_walk`.)
/// The unit exposes no group variables, so `want_path` here only rebuilds the path for a
/// path-variable caller — never binds `e`/`x`/`y` (that is a subpath sink's job).
pub(super) fn reachable_each(
    graph: &Graph,
    ctx: &Ctx,
    binding: &mut Binding,
    from: u32,
    rel: &CRel,
    spec: WalkSpec,
    on_end: OnEnd<'_>,
) -> bool {
    let unit = CUnit {
        elems: vec![CElem::Hop(CHop {
            rel: rel.clone(),
            target_slot: None,
        })],
        start_slot: None,
        where_: None,
    };
    reachable_each_unit(graph, ctx, binding, from, &unit, spec, on_end)
}

/// Collect every trail endpoint into a `Vec` (eager). For callers that genuinely
/// consume the whole set (e.g. grouped-count replay); short-circuiting consumers
/// (`EXISTS`/`LIMIT`) use `reachable_each` directly so they can stop early.
pub(super) fn reachable(
    graph: &Graph,
    ctx: &Ctx,
    from: u32,
    rel: &CRel,
    q: Quantifier,
    mode: PathMode,
) -> Vec<u32> {
    let mut ends: Vec<u32> = Vec::new();
    // Endpoint-only collector for predicate-free var-length segments (the count
    // replay path). Patterns carrying a per-hop predicate are routed to the general
    // matcher upstream, so the throwaway binding here never drives a filter.
    let mut scratch = Binding::default();
    reachable_each(
        graph,
        ctx,
        &mut scratch,
        from,
        rel,
        WalkSpec {
            q,
            mode,
            want_path: false,
        },
        &mut |_b, e, _, _, _| {
            ends.push(e);
            true
        },
    );
    ends
}

/// `ANY SHORTEST` over a single quantified segment `(start)-[rel q]->(end)`: from
/// the already-matched `seed` (bound to `start`), find one fewest-hop path to each
/// reachable vertex that matches `end`, bind it to the path variable (if named),
/// and emit. BFS gives the shortest hop distance and a predecessor tree; a vertex
/// is discovered once (its first, shortest predecessor), so one path per endpoint.
///
/// Determinism (so native == TS, byte-identical): incident edges are processed in
/// ascending global edge index — the canonical order both engines share — so the
/// predecessor chosen for each vertex is identical, and endpoints are emitted in
/// ascending vertex id. `q.max` bounds the BFS depth; `q.min ≤ 1` is enforced at
/// parse time (a larger minimum needs longer-than-shortest search).
/// Bind `end` as a walk's endpoint node, expose the reconstructed `(verts, edges)` as
/// the pattern's path variable if it names one, emit, and restore — the shared tail of
/// every path-selector walk (SHORTEST / ALL SHORTEST / ALL / ANY / SHORTEST k). Returns
/// the consumer's keep-going, so callers stop on `!emit_walk_end(..)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_walk_end(
    graph: &Graph,
    ctx: &Ctx,
    b: &mut Binding,
    end_node: &CNode,
    end: u32,
    path_slot: Option<usize>,
    verts: &[u32],
    edges: &[u32],
    emit: &mut dyn FnMut(&mut Binding) -> bool,
) -> bool {
    match_node_then(graph, ctx, b, end_node, end, &mut |b| {
        if let Some(s) = path_slot {
            b.set(s, Val::path(verts.to_vec(), edges.to_vec()));
        }
        let keep = emit(b);
        if let Some(s) = path_slot {
            b.unset(s);
        }
        keep
    })
}

/// The endpoint set for a shortest / all-shortest BFS: every vertex whose shortest
/// distance is ≥ `q.min`, plus the seed itself when `q.min ≥ 1` and a cycle returns to
/// it within the hop ceiling (`(a)-[]->+(a)`). Returns the ascending-id-sorted ends and
/// whether the seed was added as a cycle endpoint (the caller reconstructs the seed path
/// specially in that case). `seed_cycle_dist` is the shortest cycle length back to the
/// seed, or `None` if none exists; `q.min ≤ 1` is enforced upstream, so the seed added
/// here is never already present at dist 0 (the min-0 case). Shared, byte-identically,
/// by `shortest_walk` (single predecessor) and `all_shortest_walk` (all predecessors).
pub(super) fn shortest_ends(
    dist: &HashMap<u32, u32>,
    q: &Quantifier,
    seed: u32,
    seed_cycle_dist: Option<u32>,
) -> (Vec<u32>, bool) {
    let mut ends: Vec<u32> = dist
        .iter()
        .filter(|&(_, &d)| d >= q.min)
        .map(|(&v, _)| v)
        .collect();
    let seed_cycle_end =
        q.min >= 1 && seed_cycle_dist.is_some_and(|cd| q.max.is_none_or(|m| cd <= m));
    if seed_cycle_end {
        ends.push(seed);
    }
    ends.sort_unstable();
    (ends, seed_cycle_end)
}

pub(super) fn shortest_walk(
    graph: &Graph,
    ctx: &Ctx,
    pattern: &CPath,
    seed: u32,
    binding: &mut Binding,
    emit: &mut dyn FnMut(&mut Binding) -> bool,
) -> bool {
    if ctx.faulted() {
        return true;
    }
    let seg = &pattern.segments[0];
    let rel = &seg.rel;
    let end_node = &seg.node;
    let q = rel
        .quantifier
        .expect("an ANY SHORTEST pattern has a quantified segment");

    // BFS: shortest hop distance + predecessor (vertex, edge) for each vertex.
    let mut dist: HashMap<u32, u32> = HashMap::from([(seed, 0)]);
    let mut pred: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut queue: VecDeque<u32> = VecDeque::from([seed]);
    // The shortest cycle back to the seed `(dist, last-predecessor, last-edge)`:
    // the seed is marked at distance 0 and never re-discovered, so a `+`/`{1,n}`
    // path that closes on the seed (`(a)-[]->+(a)`, or any endpoint reached via a
    // cycle) would otherwise be missed. BFS order makes the first re-arrival the
    // shortest, and identical across engines.
    let mut seed_cycle: Option<(u32, u32, u32)> = None;

    while let Some(v) = queue.pop_front() {
        let d = dist[&v];
        if q.max.is_some_and(|m| d >= m) {
            continue; // don't expand past the hop ceiling
        }
        let mut nbrs: Vec<(u32, u32)> = expand_filtered(graph, ctx, binding, rel, v);
        nbrs.sort_unstable_by_key(|&(eidx, _)| eidx);
        for (eidx, nbr) in nbrs {
            if nbr == seed && seed_cycle.is_none() {
                seed_cycle = Some((d + 1, v, eidx));
            }
            if let std::collections::hash_map::Entry::Vacant(slot) = dist.entry(nbr) {
                slot.insert(d + 1);
                pred.insert(nbr, (v, eidx));
                queue.push_back(nbr);
            }
        }
    }

    // Endpoints: every vertex reached within [min, max] hops, ascending by id.
    let (ends, seed_cycle_end) = shortest_ends(&dist, &q, seed, seed_cycle.map(|(cd, _, _)| cd));

    for end in ends {
        let path = if end == seed && seed_cycle_end {
            let (_, pv, edge) = seed_cycle.expect("seed_cycle_end implies Some");
            reconstruct_cycle(seed, pv, edge, &pred)
        } else {
            reconstruct_path(seed, end, &pred)
        };
        let path_slot = pattern.path_var_slot;
        if !emit_walk_end(
            graph, ctx, binding, end_node, end, path_slot, &path.0, &path.1, emit,
        ) {
            return false;
        }
    }

    true
}

/// Walk the BFS predecessor tree back from `end` to `seed`, returning the path's
/// `(vertices, edges)` in forward order. `end == seed` gives the zero-hop path.
pub(super) fn reconstruct_path(
    seed: u32,
    end: u32,
    pred: &HashMap<u32, (u32, u32)>,
) -> (Vec<u32>, Vec<u32>) {
    let mut vertices = vec![end];
    let mut edges = Vec::new();
    let mut cur = end;
    while cur != seed {
        let (prev, edge) = pred[&cur];
        edges.push(edge);
        vertices.push(prev);
        cur = prev;
    }
    vertices.reverse();
    edges.reverse();

    (vertices, edges)
}

/// Reconstruct a shortest cycle back to the seed: the forward path `seed … pv`
/// (from the BFS predecessor tree) closed by the final edge `pv --edge--> seed`.
pub(super) fn reconstruct_cycle(
    seed: u32,
    pv: u32,
    edge: u32,
    pred: &HashMap<u32, (u32, u32)>,
) -> (Vec<u32>, Vec<u32>) {
    let (mut vertices, mut edges) = reconstruct_path(seed, pv, pred);
    vertices.push(seed);
    edges.push(edge);

    (vertices, edges)
}

/// Every shortest path `seed … end` through the shortest-path DAG `preds` (each
/// vertex → all its fewest-hop predecessors `(prev, edge)`), in forward order.
/// Deterministic: `preds` were recorded in BFS / ascending-eidx order and are
/// enumerated in that order, so native and TS produce identical path sequences.
pub(super) fn enumerate_shortest_paths(
    seed: u32,
    end: u32,
    preds: &HashMap<u32, Vec<(u32, u32)>>,
) -> Vec<(Vec<u32>, Vec<u32>)> {
    if end == seed {
        return vec![(vec![seed], Vec::new())];
    }
    let Some(ps) = preds.get(&end) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for &(pv, edge) in ps {
        for (mut vs, mut es) in enumerate_shortest_paths(seed, pv, preds) {
            vs.push(end);
            es.push(edge);
            out.push((vs, es));
        }
    }
    out
}

/// `ALL SHORTEST` over a single quantified segment: every fewest-hop path to each
/// reachable `end`-matching vertex (per the ISO selector). Like [`shortest_walk`]'s
/// BFS, but records ALL shortest predecessors per vertex (not just the first) and
/// enumerates the resulting shortest-path DAG. Determinism identical to
/// `shortest_walk` — edges in ascending eidx, endpoints ascending by id — plus the
/// per-endpoint paths in `preds`-recording order, so native == TS byte for byte.
pub(super) fn all_shortest_walk(
    graph: &Graph,
    ctx: &Ctx,
    pattern: &CPath,
    seed: u32,
    binding: &mut Binding,
    emit: &mut dyn FnMut(&mut Binding) -> bool,
) -> bool {
    if ctx.faulted() {
        return true;
    }
    let seg = &pattern.segments[0];
    let rel = &seg.rel;
    let end_node = &seg.node;
    let q = rel
        .quantifier
        .expect("an ALL SHORTEST pattern has a quantified segment");

    let mut dist: HashMap<u32, u32> = HashMap::from([(seed, 0)]);
    let mut preds: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let mut queue: VecDeque<u32> = VecDeque::from([seed]);
    // All shortest cycles back to the seed (it's never re-discovered via `dist`):
    // the min-distance edges `(prev, edge)` with `prev --edge--> seed`.
    let mut seed_cycle_dist: Option<u32> = None;
    let mut seed_cycles: Vec<(u32, u32)> = Vec::new();

    while let Some(v) = queue.pop_front() {
        let d = dist[&v];
        if q.max.is_some_and(|m| d >= m) {
            continue;
        }
        let mut nbrs: Vec<(u32, u32)> = expand_filtered(graph, ctx, binding, rel, v);
        nbrs.sort_unstable_by_key(|&(eidx, _)| eidx);
        for (eidx, nbr) in nbrs {
            if nbr == seed {
                match seed_cycle_dist {
                    None => {
                        seed_cycle_dist = Some(d + 1);
                        seed_cycles.push((v, eidx));
                    }
                    Some(cd) if cd == d + 1 => seed_cycles.push((v, eidx)),
                    _ => {}
                }
            }
            match dist.get(&nbr).copied() {
                None => {
                    dist.insert(nbr, d + 1);
                    preds.insert(nbr, vec![(v, eidx)]);
                    queue.push_back(nbr);
                }
                // Another shortest predecessor: same min distance, one hop back.
                Some(dn) if dn == d + 1 => preds.entry(nbr).or_default().push((v, eidx)),
                _ => {}
            }
        }
    }

    let (ends, seed_cycle_end) = shortest_ends(&dist, &q, seed, seed_cycle_dist);

    for end in ends {
        let paths: Vec<(Vec<u32>, Vec<u32>)> = if end == seed && seed_cycle_end {
            let mut out = Vec::new();
            for &(pv, edge) in &seed_cycles {
                for (mut vs, mut es) in enumerate_shortest_paths(seed, pv, &preds) {
                    vs.push(seed);
                    es.push(edge);
                    out.push((vs, es));
                }
            }
            out
        } else {
            enumerate_shortest_paths(seed, end, &preds)
        };
        for (vertices, edges) in paths {
            let path_slot = pattern.path_var_slot;
            if !emit_walk_end(
                graph, ctx, binding, end_node, end, path_slot, &vertices, &edges, emit,
            ) {
                return false;
            }
        }
    }
    true
}

/// Bare path binding over a single quantified segment (`p = (a)-[:R]->{m,n}(b)`):
/// enumerate every walk from the seed under the pattern's mode and bind each as a
/// Path value (vertices + edges). The plain `match_path` driver only knows the
/// endpoint, so this asks `reachable_each` for the whole walk (`want_path`).
pub(super) fn all_walk(
    graph: &Graph,
    ctx: &Ctx,
    pattern: &CPath,
    seed: u32,
    binding: &mut Binding,
    emit: &mut dyn FnMut(&mut Binding) -> bool,
) -> bool {
    let seg = &pattern.segments[0];
    let rel = &seg.rel;
    let end_node = &seg.node;
    let q = rel
        .quantifier
        .expect("bare path binding has a quantified segment");
    let path_slot = pattern.path_var_slot;
    reachable_each(
        graph,
        ctx,
        binding,
        seed,
        rel,
        WalkSpec {
            q,
            mode: pattern.mode,
            want_path: true,
        },
        &mut |b, end, verts, edges, _steps: &[StepRec]| {
            emit_walk_end(graph, ctx, b, end_node, end, path_slot, verts, edges, emit)
        },
    )
}

/// Bare `ANY` selector: one arbitrary path per endpoint — the first walk that
/// reaches each distinct endpoint in trail-discovery order. Byte-identical because
/// that order is. Built on `reachable_each`, so it honours the pattern's mode and
/// any per-hop edge predicate.
pub(super) fn any_walk(
    graph: &Graph,
    ctx: &Ctx,
    pattern: &CPath,
    seed: u32,
    binding: &mut Binding,
    emit: &mut dyn FnMut(&mut Binding) -> bool,
) -> bool {
    let seg = &pattern.segments[0];
    let rel = &seg.rel;
    let end_node = &seg.node;
    let q = rel
        .quantifier
        .expect("an ANY pattern has a quantified segment");
    let path_slot = pattern.path_var_slot;
    let mut seen: HashSet<u32> = HashSet::new();
    reachable_each(
        graph,
        ctx,
        binding,
        seed,
        rel,
        WalkSpec {
            q,
            mode: pattern.mode,
            want_path: path_slot.is_some(),
        },
        &mut |b, end, verts, edges, _steps: &[StepRec]| {
            // First witness per endpoint only (the endpoint match is per-vertex, so
            // a non-matching endpoint never emits regardless of which walk reached
            // it — marking it seen just avoids re-trying).
            if !seen.insert(end) {
                return true;
            }
            emit_walk_end(graph, ctx, b, end_node, end, path_slot, verts, edges, emit)
        },
    )
}

/// One enumerated trail to an endpoint: `(length, vertices, edges)`.
type TrailPath = (usize, Vec<u32>, Vec<u32>);

/// `SHORTEST k [GROUP]` selector: enumerate every trail, group by endpoint, order
/// each endpoint's paths by (length, trail-discovery order), then keep the first
/// `k` (plain) or every path whose length is among the `k` smallest distinct
/// lengths (`group`, the `.1` of `spec`). Byte-identical because the enumeration
/// and the stable sort are. Trades the BFS shortcut for full enumeration (needed
/// to see beyond the single shortest length); the trail budget guards a
/// pathological `*`.
pub(super) fn shortest_k_walk(
    graph: &Graph,
    ctx: &Ctx,
    pattern: &CPath,
    seed: u32,
    binding: &mut Binding,
    emit: &mut dyn FnMut(&mut Binding) -> bool,
    spec: (u32, bool),
) -> bool {
    let (k, group) = spec;
    let seg = &pattern.segments[0];
    let rel = &seg.rel;
    let end_node = &seg.node;
    let q = rel
        .quantifier
        .expect("a SHORTEST k pattern has a quantified segment");
    let path_slot = pattern.path_var_slot;

    // endpoint -> its trails as (length, vertices, edges), in discovery order.
    let mut per_end: HashMap<u32, Vec<TrailPath>> = HashMap::new();
    reachable_each(
        graph,
        ctx,
        binding,
        seed,
        rel,
        WalkSpec {
            q,
            mode: pattern.mode,
            want_path: true,
        },
        &mut |_b, end, verts, edges, _steps: &[StepRec]| {
            per_end
                .entry(end)
                .or_default()
                .push((edges.len(), verts.to_vec(), edges.to_vec()));
            true
        },
    );
    if ctx.faulted() {
        return true;
    }

    let mut ends: Vec<u32> = per_end.keys().copied().collect();
    ends.sort_unstable();

    for end in ends {
        let mut paths = per_end.remove(&end).unwrap();
        // Stable sort by length → shortest first, discovery order within a length.
        paths.sort_by_key(|(len, _, _)| *len);

        let selected: Vec<(Vec<u32>, Vec<u32>)> = if group {
            // The k smallest distinct lengths (paths are length-sorted, so equal
            // lengths are contiguous); keep every path at or below the kth.
            let mut distinct: Vec<usize> = Vec::new();
            for (len, _, _) in &paths {
                if distinct.last() != Some(len) {
                    distinct.push(*len);
                }
            }
            // The kth smallest distinct length (or the largest, if fewer than k).
            let cutoff = distinct.get((k as usize).min(distinct.len()).saturating_sub(1));
            match cutoff.copied() {
                Some(cutoff) => paths
                    .into_iter()
                    .filter(|(len, _, _)| *len <= cutoff)
                    .map(|(_, v, e)| (v, e))
                    .collect(),
                None => Vec::new(),
            }
        } else {
            paths
                .into_iter()
                .take(k as usize)
                .map(|(_, v, e)| (v, e))
                .collect()
        };

        for (vertices, edges) in selected {
            if !emit_walk_end(
                graph, ctx, binding, end_node, end, path_slot, &vertices, &edges, emit,
            ) {
                return false;
            }
        }
    }
    true
}

/// Can a selector pattern reduce to a BFS driver? True when the single
/// variable-length segment is a `*`/`+` (min ≤ 1) with no per-hop predicate — the
/// exact shape `shortest_walk`/`all_shortest_walk` are correct for. `ANY` and
/// `SHORTEST 1 [GROUP]` then reuse the O(V+E) BFS instead of enumerating trails.
pub(super) fn bfs_reducible(pattern: &CPath) -> bool {
    pattern.segments.len() == 1
        && pattern.segments[0]
            .rel
            .quantifier
            .is_some_and(|q| q.min <= 1)
        && pattern.segments[0].rel.props.is_empty()
        && pattern.segments[0].rel.where_.is_none()
}

/// Seed and match a single path pattern, emitting each binding via `emit`.
/// `where_` is the enclosing clause WHERE, threaded here only so the start node
/// can seed from a property index on a `WHERE var.k = $x` conjunct (in addition
/// to an inline `{k: $x}`); the full filter is still applied post-join.
pub(super) fn visit_pattern(
    graph: &Graph,
    ctx: &Ctx,
    pattern: &CPath,
    where_: Option<&CExpr>,
    binding: &mut Binding,
    emit: &mut dyn FnMut(&mut Binding) -> bool,
) -> bool {
    let mut at_seed = |seed: u32, binding: &mut Binding| {
        match_node_then(
            graph,
            ctx,
            binding,
            &pattern.start,
            seed,
            &mut |b| match pattern.selector {
                // A bare path variable over a single quantified segment binds each
                // enumerated walk as a Path; otherwise the plain endpoint walk.
                PathSelector::Walk if pattern.path_var_slot.is_some() => {
                    all_walk(graph, ctx, pattern, seed, b, emit)
                }
                PathSelector::Walk => match_path(graph, ctx, pattern, 0, seed, b, emit),
                // `ANY` and `SHORTEST 1 [GROUP]` over a shortest-shaped segment
                // reduce to the O(V+E) BFS drivers (a shortest path is a valid
                // arbitrary / 1-shortest path) instead of enumerating exponentially
                // many trails. Both engines route identically → still byte-identical.
                PathSelector::Any if bfs_reducible(pattern) => {
                    shortest_walk(graph, ctx, pattern, seed, b, emit)
                }
                PathSelector::Any => any_walk(graph, ctx, pattern, seed, b, emit),
                PathSelector::AnyShortest => shortest_walk(graph, ctx, pattern, seed, b, emit),
                PathSelector::AllShortest => all_shortest_walk(graph, ctx, pattern, seed, b, emit),
                PathSelector::ShortestK { k: 1, group: false } if bfs_reducible(pattern) => {
                    shortest_walk(graph, ctx, pattern, seed, b, emit)
                }
                PathSelector::ShortestK { k: 1, group: true } if bfs_reducible(pattern) => {
                    all_shortest_walk(graph, ctx, pattern, seed, b, emit)
                }
                PathSelector::ShortestK { k, group } => {
                    shortest_k_walk(graph, ctx, pattern, seed, b, emit, (k, group))
                }
            },
        )
    };
    match pattern.start.var_slot {
        // An already-bound start variable fixes the single seed.
        Some(s) if binding.bound(s) => match binding.get(s) {
            Some(Val::Node(i)) => at_seed(*i, binding),
            _ => true,
        },
        // Otherwise prefer a property-index seek (indexed inline `{k:$x}` or a
        // `WHERE this.k=$x` conjunct), falling back to the label bucket / live
        // range. Without this, a comma-joined multi-pattern MATCH bails out of
        // every vectorized (seek-capable) path and full-scans *every* anchor —
        // the O(n) footgun multi-anchor index-seed planning closes; `build_scan` already does this for the
        // single-pattern fast path. Postings are live-only in principle, but the
        // index can lag a delete, so re-check liveness (as `build_scan` does).
        //
        // Only a *named* start can carry a WHERE hint: `prop_index_hint`'s
        // slot filter treats a `None` slot as "any", so handing the clause WHERE
        // to an anonymous node (which WHERE can't even reference) would let it
        // seed on another var's conjunct. Inline props seed regardless — they're
        // this node's own.
        _ => match node_index_seed(
            graph,
            ctx,
            &pattern.start,
            pattern.start.var_slot.and(where_),
        ) {
            Some(cands) => {
                for seed in cands {
                    if graph.is_vertex_live(seed) && !at_seed(seed, binding) {
                        return false;
                    }
                }
                true
            }
            None => for_each_seed(graph, ctx, pattern.start.label.as_ref(), &mut |seed| {
                at_seed(seed, binding)
            }),
        },
    }
}

/// How much it costs to START this pattern, given what is already bound. Lower
/// is better; see [`pick_pattern`].
fn pattern_rank(p: &CPath, binding: &Binding) -> u8 {
    let bound = |s: Option<usize>| s.is_some_and(|s| binding.bound(s));
    let connected = bound(p.start.var_slot)
        || p.segments
            .iter()
            .any(|CSegment { rel, node, .. }| bound(rel.var_slot) || bound(node.var_slot));
    if connected {
        return 0; // continues an existing binding — no fresh scan at all
    }
    let restricted = p.start.label.is_some() || !p.start.props.is_empty();
    u8::from(!restricted) + 1 // 1 = restricted scan, 2 = every vertex
}

/// The next pattern to extend into, or `None` when all are done.
///
/// This is GQL's half of the rule Gremlin's `match()` has always had
/// (`pick_runnable` in `gremlin/exec.rs`): run a pattern whose start is already
/// bound before one that needs a fresh scan. Without it the join runs in the
/// order the patterns were TYPED, and two spellings of one query cost wildly
/// different amounts — a selective anchor written last is enumerated last, so
/// the unanchored pattern becomes the outer loop:
///
/// ```text
///   MATCH (x:S)-[:R]->(b), (b)-[:R]->(c) WHERE x.k = 'target'   -- constant
///   MATCH (b)-[:R]->(c), (x:S)-[:R]->(b) WHERE x.k = 'target'   -- linear
/// ```
///
/// Same single row. Measured before this function existed: 1,278x apart at 3k
/// vertices, 121,336x at 300k, and growing — the anchored spelling is constant
/// while the other scans the graph. `docs/design/query-ir.md` has the table.
///
/// Ties keep the WRITTEN order (strict `<`), so a query whose patterns are all
/// equally cheap to start runs, and emits rows, exactly as it did before.
fn pick_pattern(patterns: &[CPath], done: &[bool], binding: &Binding) -> Option<usize> {
    let mut best: Option<(u8, usize)> = None;
    for (i, p) in patterns.iter().enumerate() {
        if done[i] {
            continue;
        }
        let rank = pattern_rank(p, binding);
        if best.is_none_or(|(r, _)| rank < r) {
            best = Some((rank, i));
        }
        if rank == 0 {
            break; // can't do better than continuing an existing binding
        }
    }
    best.map(|(_, i)| i)
}

/// Extend a binding through every pattern (nested), filter by an optional WHERE,
/// and emit each surviving binding. Returns `false` if `emit` asked to stop.
///
/// Patterns are visited in the order [`pick_pattern`] chooses, not the order they
/// were written.
pub(super) fn visit_patterns(
    graph: &Graph,
    ctx: &Ctx,
    patterns: &[CPath],
    where_: Option<&CExpr>,
    binding: &mut Binding,
    emit: &mut dyn FnMut(&mut Binding) -> bool,
) -> bool {
    let mut done = vec![false; patterns.len()];
    visit_remaining(graph, ctx, patterns, &mut done, where_, binding, emit)
}

fn visit_remaining(
    graph: &Graph,
    ctx: &Ctx,
    patterns: &[CPath],
    done: &mut Vec<bool>,
    where_: Option<&CExpr>,
    binding: &mut Binding,
    emit: &mut dyn FnMut(&mut Binding) -> bool,
) -> bool {
    let Some(idx) = pick_pattern(patterns, done, binding) else {
        if let Some(w) = where_ {
            let env = Env::new(graph, ctx, binding);
            if as_truth(&eval(&env, w)) != Some(true) {
                return true; // filtered out, keep going
            }
        }
        return emit(binding);
    };
    done[idx] = true;
    let cont = visit_pattern(graph, ctx, &patterns[idx], where_, binding, &mut |b| {
        visit_remaining(graph, ctx, patterns, done, where_, b, emit)
    });
    done[idx] = false; // backtrack
    cont
}

/// Reachability fast path for `EXISTS { (a)-[:T]->+/*(b …) }`: a single unbounded
/// var-length segment from an already-bound `a` is *reachability* — BFS the reached
/// set and stop at the first vertex satisfying the endpoint (label / inline props /
/// WHERE), instead of enumerating trails (exponential — it hits the trail budget and
/// faults, e.g. testing whether an *unreachable* target is reachable). Returns
/// `Some(bool)` when it applies, else `None` (fall back to the general matcher).
/// Does a ≥1-hop `dir`-directed, `el`-labeled path `from → to` exist? BIDIRECTIONAL
/// BFS: grow a forward frontier from `from` and a backward frontier from `to`, always
/// expanding the smaller, and stop the instant they meet (`to` reached forward, `from`
/// reached backward, or the frontiers intersect) — or both exhaust. Same boolean as a
/// one-directional search, but the NEGATIVE case costs ≈ min(forward-cone, backward-cone)
/// instead of the full forward cone — the win when both endpoints are bound (a reachability
/// `EXISTS` like group-membership or resource-ancestry). Vertex-visited = reachability
/// (mode-independent, as `EXISTS` is).
pub(super) fn reachable_sets_bidir(
    graph: &Graph,
    ctx: &Ctx,
    from_set: &FxHashSet<u32>,
    to_set: &FxHashSet<u32>,
    dir: Direction,
    el: Option<&CLabelExpr>,
    min0: bool,
) -> bool {
    // A 0-hop path (var-length min-0) is a vertex shared by both ends.
    if min0 {
        let (small, big) = if from_set.len() <= to_set.len() {
            (from_set, to_set)
        } else {
            (to_set, from_set)
        };
        if small.iter().any(|v| big.contains(v)) {
            return true;
        }
    }
    let rev = flip_direction(dir);
    let n = graph.vertex_count();
    let mut vf = crate::graph::BitSet::zeros(n); // reachable FROM `from_set` (≥1 hop)
    let mut vb = crate::graph::BitSet::zeros(n); // can REACH `to_set` (≥1 hop)
    let mut ff: Vec<u32> = Vec::new();
    let mut fb: Vec<u32> = Vec::new();
    let seed = |set: &FxHashSet<u32>,
                d: Direction,
                seen: &mut crate::graph::BitSet,
                front: &mut Vec<u32>| {
        for &v in set {
            for (_e, w) in expand(graph, ctx, v, d, el) {
                if !seen.get(w as usize) {
                    seen.set(w as usize);
                    front.push(w);
                }
            }
        }
    };
    seed(from_set, dir, &mut vf, &mut ff);
    seed(to_set, rev, &mut vb, &mut fb);
    loop {
        // Meeting: a forward-reached vertex is a target (or already backward-visited), or a
        // backward-reached vertex is a source (or already forward-visited).
        if ff
            .iter()
            .any(|&w| to_set.contains(&w) || vb.get(w as usize))
        {
            return true;
        }
        if fb
            .iter()
            .any(|&w| from_set.contains(&w) || vf.get(w as usize))
        {
            return true;
        }
        // Terminate as soon as EITHER side is fully explored without meeting — no need to
        // exhaust the other, possibly huge, cone (the negative-case win).
        if ff.is_empty() || fb.is_empty() {
            return false;
        }
        // Expand the smaller frontier.
        let forward = ff.len() <= fb.len();
        let (cur, d, seen, front) = if forward {
            (std::mem::take(&mut ff), dir, &mut vf, &mut ff)
        } else {
            (std::mem::take(&mut fb), rev, &mut vb, &mut fb)
        };
        for u in cur {
            for (_e, w) in expand(graph, ctx, u, d, el) {
                if !seen.get(w as usize) {
                    seen.set(w as usize);
                    front.push(w);
                }
            }
        }
    }
}

/// Closed forward DFS from `sources` over `dir`/`el`: each vertex is discovered and
/// expanded exactly once (dedup via a `BitSet`), the source vertices themselves NOT
/// delivered (≥1 hop only). `on_reach(w)` fires once per newly-discovered `w` in stack
/// pop-order; returning `false` short-circuits the whole walk and the helper returns
/// `false`. Returns `true` when the walk ran to exhaustion. The shared engine behind
/// existence short-circuit (`any_match_reachable`) and distinct-set collection
/// (`try_reachable_distinct`); the ≥1-hop discovery order is load-bearing for both.
pub(super) fn reach_dfs_forward<F: FnMut(u32) -> bool>(
    graph: &Graph,
    ctx: &Ctx,
    sources: impl IntoIterator<Item = u32>,
    dir: Direction,
    el: Option<&CLabelExpr>,
    mut on_reach: F,
) -> bool {
    let mut seen = crate::graph::BitSet::zeros(graph.vertex_count());
    let mut stack: Vec<u32> = Vec::new();
    let discover = |w: u32, seen: &mut crate::graph::BitSet, stack: &mut Vec<u32>| -> bool {
        !seen.get(w as usize) && {
            seen.set(w as usize);
            stack.push(w);
            true
        }
    };
    for s in sources {
        for (_e, w) in expand(graph, ctx, s, dir, el) {
            if discover(w, &mut seen, &mut stack) && !on_reach(w) {
                return false;
            }
        }
    }
    while let Some(u) = stack.pop() {
        for (_e, w) in expand(graph, ctx, u, dir, el) {
            if discover(w, &mut seen, &mut stack) && !on_reach(w) {
                return false;
            }
        }
    }
    true
}

pub(super) fn any_match_reachable(
    graph: &Graph,
    ctx: &Ctx,
    patterns: &[CPath],
    where_: Option<&CExpr>,
    binding: &Binding,
    sub_len: usize,
) -> Option<bool> {
    let [path] = patterns else { return None };
    let [seg] = path.segments.as_slice() else {
        return None;
    };
    let q = seg.rel.quantifier?;
    if q.max.is_some()
        || seg.rel.var_slot.is_some()
        || !seg.rel.props.is_empty()
        || seg.rel.where_.is_some()
        || !matches!(seg.rel.direction, Direction::Out | Direction::In)
        || !path.start.props.is_empty()
        || path.start.where_.is_some()
    {
        return None;
    }
    // The start must already be bound (the correlated `a`).
    let sv = match path.start.var_slot.and_then(|s| binding.get(s)) {
        Some(Val::Node(v)) => *v,
        _ => return None,
    };

    let mut work = binding.clone();
    work.resize(sub_len);
    let b_slot = seg.node.var_slot;
    // If the endpoint variable is *already bound* — a back-reference: the closed
    // cyclic `(a)-[:R]->+(a)`, or a second already-correlated var — then a valid
    // match must reach *that specific vertex*, not merely any reachable one.
    // Without this guard the BFS answers "does `a` reach anything" instead of
    // "does `a` reach the target", so e.g. every DAG vertex wrongly looks on-cycle.
    let bound_end: Option<u32> = match b_slot.and_then(|bs| binding.get(bs)) {
        Some(Val::Node(v)) => Some(*v),
        Some(_) => return None, // bound to a non-node: decline to the general matcher
        None => None,
    };
    // Is `v` a valid endpoint `b` (bound-target + label + inline props/WHERE + the
    // EXISTS WHERE)?
    let hit = |graph: &Graph, v: u32, work: &mut Binding| -> bool {
        if bound_end.is_some_and(|be| v != be) {
            return false;
        }
        if !matches_label(graph, ctx, v, seg.node.label.as_ref()) {
            return false;
        }
        if let Some(bs) = b_slot {
            work.set(bs, Val::Node(v));
        }
        if !satisfies(
            graph,
            ctx,
            &Val::Node(v),
            &seg.node.props,
            seg.node.where_.as_ref(),
            work,
        ) {
            return false;
        }
        where_.is_none_or(|w| as_truth(&eval(&Env::new(graph, ctx, work), w)) == Some(true))
    };

    // `->*` also admits the zero-length path — the start itself.
    if q.min == 0 && hit(graph, sv, &mut work) {
        return Some(true);
    }
    let (dir, el) = (seg.rel.direction, seg.rel.label.as_ref());
    // (A bound-both-endpoints single segment is handled by the unified `exists_bidir`
    // meet-in-the-middle, tried before this in `any_match`; here `bound_end` only survives
    // when that path declined — e.g. an inner WHERE — so fall to the forward search.)
    // `hit` returning true = match found → short-circuit (`on_reach` returns false).
    let exhausted = reach_dfs_forward(graph, ctx, std::iter::once(sv), dir, el, |w| {
        !hit(graph, w, &mut work)
    });
    Some(!exhausted)
}

/// Does the (correlated) sub-pattern have at least one match? Short-circuits.
/// The work binding is the outer binding grown to the sub-scope (`sub_len`):
/// outer slots stay set (correlation), the sub's own slots start unbound.
/// Resolve a relationship label expression to the concrete set of edge-type ids it can
/// match — `Some(ids)` for a single `Label` or an `Or` of labels, `None` for anything the
/// bidirectional evaluator won't reason about disjointness for (wildcard / And / Not /
/// absent label). Used only to prove two segments can't share an edge.
pub(super) fn label_etypes(el: Option<&CLabelExpr>, ctx: &Ctx) -> Option<Vec<u32>> {
    fn collect(e: &CLabelExpr, ctx: &Ctx, out: &mut Vec<u32>) -> bool {
        match e {
            CLabelExpr::Label(r) => {
                if let Some(id) = ctx.labels[*r].1 {
                    out.push(id);
                }
                true
            }
            CLabelExpr::Or(a, b) => collect(a, ctx, out) && collect(b, ctx, out),
            _ => false,
        }
    }
    let e = el?;
    let mut out = Vec::new();
    collect(e, ctx, &mut out).then_some(out)
}

/// One segment of a linear pattern reduced to a set-reachability step: its edge label,
/// direction, and whether it is a var-length closure (and admits the zero-length path).
pub(super) struct BidirStep<'a> {
    label: Option<&'a CLabelExpr>,
    dir: Direction,
    varlen: bool,
    min0: bool,
}

/// Advance a frontier vertex-set by one step — a fixed 1-hop expansion, or a full closure
/// for a var-length segment (`min0` folds in the zero-length path = the set itself).
/// `reverse` walks the step backward (for the right frontier growing from the bound target).
pub(super) fn bidir_apply(
    graph: &Graph,
    ctx: &Ctx,
    set: &FxHashSet<u32>,
    step: &BidirStep,
    reverse: bool,
) -> FxHashSet<u32> {
    let dir = if reverse {
        flip_direction(step.dir)
    } else {
        step.dir
    };
    let mut out: FxHashSet<u32> = FxHashSet::default();
    if step.varlen {
        // Every vertex reachable in ≥1 hop from `set` (BFS), plus `set` itself when min-0.
        let mut work: Vec<u32> = set.iter().copied().collect();
        while let Some(v) = work.pop() {
            for (_e, w) in expand(graph, ctx, v, dir, step.label) {
                if out.insert(w) {
                    work.push(w);
                }
            }
        }
        if step.min0 {
            out.extend(set.iter().copied());
        }
    } else {
        for &v in set {
            for (_e, w) in expand(graph, ctx, v, dir, step.label) {
                out.insert(w);
            }
        }
    }
    out
}

/// Meet-in-the-middle EXISTENCE for a LINEAR pattern (`k ≥ 1` segments) with BOTH endpoints
/// already bound (`u = n0 —seg1→ n1 … —segk→ nk = t`). Grow a left frontier-set from `u`
/// and a right frontier-set from `t`, always advancing the SMALLER by a whole segment until
/// they are ONE segment apart, then answer whether that gap segment connects them — a fixed
/// hop by expand-and-intersect, a var-length gap by an EDGE-level bidirectional reachability
/// between the sets ([`reachable_sets_bidir`]). Never expands the far, possibly huge, cone
/// (the negative-check win). **This is the single meet-in-the-middle path: a single segment
/// (`k = 1`) is the degenerate case — no advance, just the edge-level bidirectional between
/// `{u}` and `{t}`.** Existence → boolean, so byte-identical to a forward walk. Applies only
/// where reachability == existence: WALK/TRAIL mode (SIMPLE/ACYCLIC need node-distinctness
/// the set search doesn't track) with edge-type-DISJOINT segments (so no edge is reused
/// across segments under TRAIL). Declines (`None`) on anything richer — a path variable, an
/// inner WHERE, a shortest selector, a filtered intermediate node, a bounded `{m,n}`
/// quantifier, a parenthesized subpath, an edge variable/predicate, an undirected/
/// unresolvable label — falling back to `any_match_reachable` (unbound-end / inner-WHERE
/// single-segment) or the general matcher.
pub(super) fn exists_bidir(
    graph: &Graph,
    ctx: &Ctx,
    patterns: &[CPath],
    where_: Option<&CExpr>,
    binding: &Binding,
) -> Option<bool> {
    let [path] = patterns else {
        return None;
    };
    if path.segments.is_empty()
        || !matches!(path.mode, PathMode::Walk | PathMode::Trail)
        || !matches!(path.selector, PathSelector::Walk)
        || path.path_var_slot.is_some()
        || where_.is_some()
    {
        return None;
    }
    // Both endpoints already bound (correlated from the outer scope).
    let u = match path.start.var_slot.and_then(|s| binding.get(s)) {
        Some(Val::Node(v)) => *v,
        _ => return None,
    };
    let last = path.segments.last()?;
    let t = match last.node.var_slot.and_then(|s| binding.get(s)) {
        Some(Val::Node(v)) => *v,
        _ => return None,
    };
    // Endpoint nodes' own label / props / WHERE must hold (validate once); intermediate
    // nodes must be PLAIN (a filter there would restrict the frontier).
    let endpoint_ok = |v: u32, n: &CNode| -> bool {
        matches_label(graph, ctx, v, n.label.as_ref())
            && satisfies(
                graph,
                ctx,
                &Val::Node(v),
                &n.props,
                n.where_.as_ref(),
                binding,
            )
    };
    if !endpoint_ok(u, &path.start) || !endpoint_ok(t, &last.node) {
        return Some(false);
    }
    for seg in &path.segments[..path.segments.len() - 1] {
        let n = &seg.node;
        if n.label.is_some() || !n.props.is_empty() || n.where_.is_some() {
            return None;
        }
    }
    // Reduce each segment to a set-reachability step; bail on anything not handled.
    let mut steps: Vec<BidirStep> = Vec::with_capacity(path.segments.len());
    let mut etypes: Vec<Vec<u32>> = Vec::with_capacity(path.segments.len());
    for seg in &path.segments {
        let r = &seg.rel;
        if seg.unit.is_some()
            || r.var_slot.is_some()
            || !r.props.is_empty()
            || r.where_.is_some()
            || !matches!(r.direction, Direction::Out | Direction::In)
        {
            return None;
        }
        let (varlen, min0) = match r.quantifier {
            None => (false, false),
            Some(q) if q.max.is_none() => (true, q.min == 0),
            _ => return None, // bounded {m,n} needs level tracking — decline
        };
        etypes.push(label_etypes(r.label.as_ref(), ctx)?);
        steps.push(BidirStep {
            label: r.label.as_ref(),
            dir: r.direction,
            varlen,
            min0,
        });
    }
    // Edge-type-DISJOINT segments ⇒ no edge is shared across segments, so a reachable
    // (walk) match is also a TRAIL. Any overlap → decline (the set search can't prove the
    // per-edge-uniqueness a TRAIL requires).
    for i in 0..etypes.len() {
        for j in (i + 1)..etypes.len() {
            if etypes[i].iter().any(|a| etypes[j].contains(a)) {
                return None;
            }
        }
    }

    // Advance a left set from `u` and a right set from `t`, always growing the SMALLER by a
    // whole segment (forward left, reversed right), until they are ONE segment apart. Then
    // answer whether that gap segment connects the two frontiers — a fixed hop is a 1-hop
    // expand-and-intersect; a var-length gap is an EDGE-level bidirectional reachability
    // between the sets. For a single segment (`k == 1`) the loop never runs and this reduces
    // to a bound-both-ends single-segment bidirectional reachability.
    let k = steps.len();
    let mut left: FxHashSet<u32> = FxHashSet::from_iter([u]);
    let mut right: FxHashSet<u32> = FxHashSet::from_iter([t]);
    let (mut li, mut ri) = (0usize, k);
    while ri - li > 1 {
        if left.len() <= right.len() {
            left = bidir_apply(graph, ctx, &left, &steps[li], false);
            li += 1;
        } else {
            right = bidir_apply(graph, ctx, &right, &steps[ri - 1], true);
            ri -= 1;
        }
        if left.is_empty() || right.is_empty() {
            return Some(false);
        }
    }
    let intersects = |a: &FxHashSet<u32>, b: &FxHashSet<u32>| -> bool {
        let (small, big) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        small.iter().any(|v| big.contains(v))
    };
    let gap = &steps[li]; // ri == li + 1 → the segment between them
    if gap.varlen {
        Some(reachable_sets_bidir(
            graph, ctx, &left, &right, gap.dir, gap.label, gap.min0,
        ))
    } else if left.len() <= right.len() {
        Some(intersects(
            &bidir_apply(graph, ctx, &left, gap, false),
            &right,
        ))
    } else {
        Some(intersects(
            &left,
            &bidir_apply(graph, ctx, &right, gap, true),
        ))
    }
}
