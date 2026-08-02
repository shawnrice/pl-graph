//! The shared access path: what a query language has learned about ONE element
//! variable, and how to turn that into a set of candidate element ids.
//!
//! Both query engines were recognizing index seeks independently — GQL over
//! `CExpr` in `gql/eval/scan.rs`, Gremlin over `&[Step]` in `gremlin/exec.rs` —
//! and both were computing the same thing: `Option<Vec<u32>>`. Eight seeking
//! gaps have been found across the two, every one of them the same bug (a
//! recogniser keyed on the SURFACE SHAPE of a predicate, so each new spelling
//! needs its own arm), and the two had already drifted apart on which seekable
//! conjunct to prefer. See `docs/design/query-ir.md`.
//!
//! This module is the half that does not depend on either language. A front end
//! lowers its own syntax into [`ElementSeek`] — that part stays per-language,
//! and it is small — and everything below that line happens once, here.
//!
//! # Why the operand is not an `IdxKey`
//!
//! A seek on `$x` cannot be fully resolved when the query is planned, because
//! the value is not known yet. But the SHAPE can be: which key, which operator,
//! and whether the value is a literal or a parameter slot. That split is the
//! whole point of the design —
//!
//!   - **plan time** builds an [`ElementSeek`], which is where normalization
//!     lives and where the expensive pattern matching happens ONCE;
//!   - **execution** calls [`ElementSeek::resolve`], which binds parameters and
//!     performs exactly one seek.
//!
//! Recognition used to happen per execution, and a traversal pattern ran it
//! three times over — each doing a full index seek and discarding two of the
//! three results, at a cost proportional to the seed size.

use std::collections::HashSet;
use std::sync::Arc;

use crate::graph::{Column, Graph, IdxKey, RangeBound};

/// How a parameter slot resolves to index keys.
///
/// Two methods rather than one because `IN $p` needs the whole LIST, and a
/// scalar-only resolver would quietly treat a bound list as a single value —
/// seeking one wrong key instead of unioning several right ones. `None` from
/// either means "no index can answer this", which makes the predicate
/// unseekable rather than wrong.
pub trait Bindings {
    fn scalar(&self, slot: usize) -> Option<IdxKey>;

    /// A slot holding a list. The default treats a scalar as a one-element
    /// list, matching `IN $p` where `$p` was bound to a single value.
    fn list(&self, slot: usize) -> Option<Vec<IdxKey>> {
        self.scalar(slot).map(|k| vec![k])
    }
}

/// So a test (or any front end with no list params) can pass a plain closure.
impl<F: Fn(usize) -> Option<IdxKey>> Bindings for F {
    fn scalar(&self, slot: usize) -> Option<IdxKey> {
        self(slot)
    }
}

/// A comparison a property index can answer. Deliberately NOT the parser's
/// operator set: `<>` cannot seek (it is the complement of a point, which is
/// most of the index), and neither can `IS NULL`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SeekOp {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

impl SeekOp {
    /// `a OP b` as `b OP' a` — the same predicate with the operands exchanged,
    /// so `$x < u.n` normalizes to `u.n > $x` rather than being missed.
    /// Equality is symmetric; the orderings invert.
    #[must_use]
    pub fn flipped(self) -> Self {
        match self {
            Self::Lt => Self::Gt,
            Self::Le => Self::Ge,
            Self::Gt => Self::Lt,
            Self::Ge => Self::Le,
            Self::Eq => Self::Eq,
        }
    }
}

/// Where a compared value comes from. `Param` is resolved at execution.
///
/// `PartialEq` is what lets a test assert that two spellings collapsed to the
/// SAME structure, which is a stronger statement than their costing the same.
#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Lit(IdxKey),
    Param(usize),
}

impl Operand {
    fn resolve(&self, param: &impl Bindings) -> Option<IdxKey> {
        match self {
            Self::Lit(k) => Some(k.clone()),
            Self::Param(slot) => param.scalar(*slot),
        }
    }
}

/// A resolved seek: the candidate ids, and which key they answer exactly.
#[derive(Clone, Debug)]
pub struct Seeded {
    pub ids: Vec<u32>,
    /// The key whose constraints the seed satisfies exactly, so a caller may
    /// drop the filter that produced them. `None` ⇒ treat the seed as a
    /// superset and re-apply everything.
    pub exact_key: Option<Arc<str>>,
}

/// One branch of an `OR` — its own conjunction of predicates.
pub type Branch = Vec<KeyPredicate>;
/// One `OR`: a union over branches. Named because the alternative is a
/// `Vec<Vec<Vec<KeyPredicate>>>` whose levels nobody can keep straight.
#[derive(Clone, Debug, PartialEq)]
pub enum Disjunction {
    /// Branches known when the query was planned.
    Branches(Vec<Branch>),
    /// `key IN $p`, where `$p` holds a list. The VALUES are not known at plan
    /// time and neither is how many there are, so this expands to one equality
    /// branch per element at resolve time. Without it, an `IN` over a parameter
    /// could only ever be recognized during execution, which is the thing this
    /// layer exists to stop doing.
    AnyOfParam { key: Arc<str>, slot: usize },
}

/// One `key OP value` a front end recognized.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyPredicate {
    /// Property name, dotted for a nested path (`meta.city`) — the same string
    /// the index was declared under.
    pub key: Arc<str>,
    pub op: SeekOp,
    pub operand: Operand,
}

/// Everything a front end has learned about ONE element variable.
///
/// Build it with [`ElementSeek::node`] / [`ElementSeek::edge`] and the `push_*`
/// methods, which normalize as they go so equivalent spellings become the same
/// structure. The recogniser above them never has to know that `u.k = $a OR
/// u.k = $b` and `u.k IN [$a, $b]` are the same predicate.
#[derive(Clone, Debug, PartialEq)]
pub struct ElementSeek {
    /// Which store to seek: edges have their own indexes and key ids.
    pub edge: bool,
    /// Conjunctive — every one must hold, so any single one bounds a candidate
    /// SUPERSET and the caller re-verifies. That is what lets the most selective
    /// one be chosen freely.
    conj: Vec<KeyPredicate>,
    /// Disjunctive: outer is `OR`, each inner branch is that branch's own
    /// conjunction. `IN` lists lower to one single-predicate branch per value,
    /// and an `OR` whose branches address DIFFERENT keys (`a.x = 1 OR a.y = 2`)
    /// is the same structure — both are a union over the same element space.
    ///
    /// Every branch must be seekable or that disjunction is not: missing a
    /// branch loses rows, unlike missing a conjunct. Several disjunctions can
    /// coexist — `(a OR b) AND (c OR d)` — and each is independently a valid
    /// candidate superset, so they compete on size like the conjuncts do.
    disj: Vec<Disjunction>,
}

impl ElementSeek {
    #[must_use]
    pub fn node() -> Self {
        Self {
            edge: false,
            conj: Vec::new(),
            disj: Vec::new(),
        }
    }

    #[must_use]
    pub fn edge() -> Self {
        Self {
            edge: true,
            conj: Vec::new(),
            disj: Vec::new(),
        }
    }

    /// A seek over the same store as `edge` says.
    #[must_use]
    pub fn same_kind(edge: bool) -> Self {
        if edge {
            Self::edge()
        } else {
            Self::node()
        }
    }

    /// True when nothing seekable was recognized — the caller should scan.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conj.is_empty() && self.disj.is_empty()
    }

    /// A `key OP value` conjunct.
    pub fn push(&mut self, key: Arc<str>, op: SeekOp, operand: Operand) {
        self.conj_push(KeyPredicate { key, op, operand });
    }

    /// An already-built conjunct.
    pub fn conj_push(&mut self, p: KeyPredicate) {
        self.conj.push(p);
    }

    /// This seek as the branches of an enclosing `OR`, or `None` if it cannot be
    /// expressed as one — which makes the enclosing disjunction unseekable.
    ///
    /// A plain conjunction is ONE branch. A lone disjunction flattens into the
    /// enclosing one, so `(a OR b) OR c` becomes `a OR b OR c` rather than a
    /// union containing a union. Anything else (a conjunction mixed with a
    /// disjunction) would need a distributive expansion that can blow up in
    /// size, and the honest answer there is to scan.
    #[must_use]
    pub fn into_branches(self) -> Option<Vec<Branch>> {
        match (self.conj.is_empty(), self.disj.len()) {
            (false, 0) => Some(vec![self.conj]),
            (true, 1) => match self.disj.into_iter().next()? {
                Disjunction::Branches(b) => Some(b),
                // A symbolic `IN $p` cannot be flattened without its values.
                Disjunction::AnyOfParam { .. } => None,
            },
            _ => None,
        }
    }

    /// A disjunction of equalities on ONE key — an `IN` list, or an `OR` of
    /// equalities the front end folded together.
    pub fn push_any_of(&mut self, key: Arc<str>, values: Vec<Operand>) {
        self.push_branches(
            values
                .into_iter()
                .map(|v| {
                    vec![KeyPredicate {
                        key: key.clone(),
                        op: SeekOp::Eq,
                        operand: v,
                    }]
                })
                .collect(),
        );
    }

    /// A general disjunction: outer `OR`, each branch its own conjunction.
    ///
    /// A single branch collapses into the conjunction: `u.k IN [$a]` IS
    /// `u.k = $a`, and leaving it as a one-branch union would take a different
    /// code path for a predicate that is character-for-character equivalent.
    /// That is precisely the class of divergence this module exists to remove —
    /// after this, the two spellings are the same `ElementSeek`, not merely two
    /// structures that happen to cost the same.
    ///
    /// An EMPTY disjunction is not a no-op — `u.k IN []` matches nothing — so it
    /// is kept, and resolves to an empty candidate set rather than to a scan.
    pub fn push_branches(&mut self, branches: Vec<Branch>) {
        if let [only] = branches.as_slice() {
            self.conj.extend(only.iter().cloned());
        } else {
            self.disj.push(Disjunction::Branches(branches));
        }
    }

    /// `key IN $slot`, where the parameter holds the list. Expanded at resolve.
    pub fn push_any_of_param(&mut self, key: Arc<str>, slot: usize) {
        self.disj.push(Disjunction::AnyOfParam { key, slot });
    }

    /// Candidate element ids, or `None` to scan.
    ///
    /// `param` resolves a parameter slot to an index key; it returns `None` for
    /// a value no index can answer (a list, a map, a type that was never
    /// interned), which makes that predicate unseekable rather than wrong.
    ///
    /// The result is a SUPERSET for conjunctions — the caller still applies the
    /// full predicate — and exact for a lone disjunction.
    #[must_use]
    pub fn resolve(&self, graph: &Graph, param: &impl Bindings) -> Option<Vec<u32>> {
        self.resolve_seeded(graph, param).map(|s| s.ids)
    }

    /// Would [`resolve`](Self::resolve) return a seed — WITHOUT building one.
    ///
    /// Several callers only need the boolean: the orientation chooser asks
    /// whether each end of a pattern is seekable so it can lead with the better
    /// one, and the LIMIT logic asks whether a scan became a seek. Each of those
    /// used to call the full seek and drop the ids on the floor. A traversal
    /// pattern therefore recognized and SEEKED three times and used one result,
    /// at a cost proportional to the seed — a range returning 10k ids built 30k.
    #[must_use]
    pub fn can_seek(&self, graph: &Graph, param: &impl Bindings) -> bool {
        let indexed = |k: &str| idx_indexed(graph, k, self.edge);

        ranges(&self.conj, param)
            .iter()
            .any(|(key, _)| indexed(key))
            || self.disj.iter().any(|d| match d {
                Disjunction::AnyOfParam { key, slot } => {
                    indexed(key) && param.list(*slot).is_some()
                }
                Disjunction::Branches(branches) => branches
                    .iter()
                    .all(|b| ranges(b, param).iter().any(|(key, _)| indexed(key))),
            })
    }

    /// Whether every conjunct can be answered straight from a typed column.
    ///
    /// Only then is [`scan`](Self::scan) equivalent to running the front end's
    /// own filters — see [`column_matches`] for why the types must line up.
    #[must_use]
    pub fn columnar(&self, graph: &Graph, param: &impl Bindings) -> bool {
        self.disj.is_empty()
            && !self.conj.is_empty()
            && self.conj.iter().all(|p| {
                p.operand
                    .resolve(param)
                    .and_then(|k| column_matches(graph, &p.key, self.edge, p.op, &k))
                    .is_some()
            })
    }

    /// Whether every predicate compares types that can meaningfully be compared.
    ///
    /// `false` means DO NOT SEEK — see [`type_agrees`]. A seek would return the
    /// right rows while suppressing a type fault the per-row path raises.
    #[must_use]
    pub fn types_agree(&self, graph: &Graph, param: &impl Bindings) -> bool {
        let ok = |p: &KeyPredicate| {
            p.operand
                .resolve(param)
                .is_none_or(|k| type_agrees(graph, &p.key, self.edge, &k))
        };

        self.conj.iter().all(ok)
            && self.disj.iter().all(|d| match d {
                Disjunction::Branches(b) => b.iter().flatten().all(ok),
                Disjunction::AnyOfParam { .. } => true,
            })
    }

    /// Whether every conjunct on `key` is answerable from a typed column.
    ///
    /// A caller that DROPS the filter the index answered must check this first.
    /// A seek can return the right rows while the equivalent comparison would
    /// have FAULTED — `has('k', gt(5))` on a string column is a type error in
    /// Gremlin, but the range seek just yields nothing. Dropping the step then
    /// loses the error, and the same query answers `0` with an index and throws
    /// without one, which makes an index observable instead of a pure
    /// optimization.
    #[must_use]
    pub fn answers_exactly(&self, graph: &Graph, param: &impl Bindings, key: &str) -> bool {
        self.conj.iter().filter(|p| &*p.key == key).all(|p| {
            p.operand
                .resolve(param)
                .and_then(|k| column_matches(graph, &p.key, self.edge, p.op, &k))
                .is_some()
        })
    }

    /// Every candidate satisfying EVERY conjunct: seeded from an index when one
    /// applies, otherwise from `universe`, then filtered against the typed
    /// columns.
    ///
    /// This is the part both engines can share ABOVE the seek. GQL reaches the
    /// same answer through its columnar scan and Gremlin through a per-traverser
    /// walk, and the walk measured 9-24x slower for identical work. One
    /// implementation here means an optimization lands on both at once, which is
    /// the point of the shared layer.
    ///
    /// Call only when [`columnar`](Self::columnar) holds.
    #[must_use]
    pub fn scan(
        &self,
        graph: &Graph,
        param: &impl Bindings,
        universe: impl FnOnce() -> Vec<u32>,
    ) -> Vec<u32> {
        let mut ids = self.resolve(graph, param).unwrap_or_else(universe);

        for p in &self.conj {
            let Some(test) = p
                .operand
                .resolve(param)
                .and_then(|k| column_matches(graph, &p.key, self.edge, p.op, &k))
            else {
                continue;
            };

            ids.retain(|&row| test(row));
        }

        ids
    }

    /// As [`resolve`](Self::resolve), plus which key (if any) the seed answers
    /// EXACTLY rather than as a superset.
    ///
    /// A caller that re-applies the full predicate — GQL does — can ignore it.
    /// One that drops the filter the index satisfied needs it: re-filtering a
    /// large range seed is a second pass over the whole seed, which measured a
    /// clean 2x on `range` and `between` in `gremlin_index_bench`.
    ///
    /// Conservative by construction. A conjunction seeks one key with every
    /// bound on that key folded in, so that key is exact. A disjunction is exact
    /// only when every branch is a single predicate on one shared key — anything
    /// else may have taken a per-branch superset, and calling it exact would
    /// drop a filter that is still doing work.
    #[must_use]
    pub fn resolve_seeded(&self, graph: &Graph, param: &impl Bindings) -> Option<Seeded> {
        // Take the SMALLEST candidate set among the seekable options rather than
        // the first. A conjunction ANDs necessary conditions, so every candidate
        // set is a valid superset and the smallest is the cheapest correct one.
        // Picking the first is what made `has('country','US').has('ssn',$x)` seed
        // from the wrong side in Gremlin while GQL seeded from the right one.
        //
        // Intersecting them instead was measured to LOSE: building and probing
        // two ~200k halves costs more than the scan it replaces.
        let mut best: Option<Seeded> = None;
        let mut keep = |ids: Vec<u32>, exact_key: Option<Arc<str>>| {
            if best.as_ref().is_none_or(|b| ids.len() < b.ids.len()) {
                best = Some(Seeded { ids, exact_key });
            }
        };

        for (key, rb) in ranges(&self.conj, param) {
            if let Some(ids) = seek_bound(graph, &key, &rb, self.edge) {
                keep(ids, Some(key));
            }
        }

        for d in &self.disj {
            if let Some(ids) = self.union(graph, d, param) {
                keep(ids, one_key_of(d));
            }
        }

        best
    }

    /// The best candidate set for ONE conjunction — the smallest seekable one,
    /// or `None` if nothing in it can seek.
    fn best_of(
        &self,
        graph: &Graph,
        preds: &[KeyPredicate],
        param: &impl Bindings,
    ) -> Option<Vec<u32>> {
        ranges(preds, param)
            .into_iter()
            .filter_map(|(key, rb)| seek_bound(graph, &key, &rb, self.edge))
            .min_by_key(Vec::len)
    }

    /// A disjunction seeds only when EVERY branch does — one unseekable branch
    /// means rows outside the union, so the union is no longer a superset.
    fn union(&self, graph: &Graph, d: &Disjunction, param: &impl Bindings) -> Option<Vec<u32>> {
        let owned: Vec<Branch>;
        let branches: &[Branch] = match d {
            Disjunction::Branches(b) => b,
            Disjunction::AnyOfParam { key, slot } => {
                owned = param
                    .list(*slot)?
                    .into_iter()
                    .map(|k| {
                        vec![KeyPredicate {
                            key: key.clone(),
                            op: SeekOp::Eq,
                            operand: Operand::Lit(k),
                        }]
                    })
                    .collect();
                &owned
            }
        };
        let mut seen: HashSet<u32> = HashSet::new();
        let mut out = Vec::new();

        for branch in branches {
            // Branches overlap freely, and a repeated value is a repeated
            // branch: `IN ['a','a']` is `= 'a'`. The seed is a candidate LIST,
            // so a duplicate becomes a duplicate ROW — a wrong answer, not just
            // wasted work.
            for id in self.best_of(graph, branch, param)? {
                if seen.insert(id) {
                    out.push(id);
                }
            }
        }

        Some(out)
    }
}

/// Conjuncts folded into one tight [`RangeBound`] per key.
///
/// `x >= 5 AND x <= 9` is one bounded seek, not two; and `5 <= x AND 9 >= x`
/// arrives as the same thing, because the operand order was normalized away
/// before it ever reached here.
fn ranges(preds: &[KeyPredicate], param: &impl Bindings) -> Vec<(Arc<str>, RangeBound)> {
    let mut out: Vec<(Arc<str>, RangeBound)> = Vec::new();

    {
        for p in preds {
            let Some(key) = p.operand.resolve(param) else {
                continue; // unbindable value ⇒ this conjunct simply cannot seek
            };
            let slot = match out.iter_mut().find(|(k, _)| *k == p.key) {
                Some(s) => s,
                None => {
                    out.push((p.key.clone(), RangeBound::default()));
                    out.last_mut().expect("just pushed")
                }
            };

            apply(&mut slot.1, p.op, key);
        }
    }

    out
}

/// The single key a disjunction constrains, when every branch is one predicate
/// on that same key — an `IN` list or a folded `OR` of equalities. `None` for a
/// branch that constrains several keys, where the union may be a superset.
fn one_key_of(d: &Disjunction) -> Option<Arc<str>> {
    match d {
        Disjunction::AnyOfParam { key, .. } => Some(key.clone()),
        Disjunction::Branches(branches) => {
            let mut keys = branches.iter().map(|b| match b.as_slice() {
                [only] => Some(&only.key),
                _ => None,
            });
            let first = keys.next()??.clone();

            keys.all(|k| k == Some(&first)).then_some(first)
        }
    }
}

/// Fold one comparison into an accumulating range.
fn apply(rb: &mut RangeBound, op: SeekOp, key: IdxKey) {
    match op {
        SeekOp::Eq => {
            rb.gte = Some(key.clone());
            rb.lte = Some(key);
        }
        SeekOp::Lt => rb.lt = Some(key),
        SeekOp::Le => rb.lte = Some(key),
        SeekOp::Gt => rb.gt = Some(key),
        SeekOp::Ge => rb.gte = Some(key),
    }
}

fn idx_indexed(graph: &Graph, name: &str, edge: bool) -> bool {
    if edge {
        graph.edge_indexed(name)
    } else {
        graph.vertex_indexed(name)
    }
}

/// Do the operand's type and the column's type agree?
///
/// Distinct from [`column_matches`], which also asks whether this layer can
/// EVALUATE the comparison. Here the question is only whether comparing them is
/// meaningful at all — because when it is not, the seek must be declined
/// entirely rather than answered.
///
/// A range seek for a number against an all-string column returns an empty set,
/// which is the right ROWS but skips the type FAULT that comparing them per-row
/// would raise. The query then answers `0` with an index and throws without one,
/// making the index observable. `Mixed` disagrees with everything: its rows can
/// be of any type, so some of them would fault.
fn type_agrees(graph: &Graph, key: &str, edge: bool, want: &IdxKey) -> bool {
    let store = if edge {
        &graph.edge_props
    } else {
        &graph.props
    };
    let Some(col) = store.keys.get(key).and_then(|k| store.cols.get(k as usize)) else {
        // No such key: nothing to compare against, so nothing can fault.
        return true;
    };

    matches!(
        (col, want),
        (Column::Num { .. }, IdxKey::Num(_))
            | (Column::Str { .. }, IdxKey::Str(_))
            | (Column::Bool { .. }, IdxKey::Bool(_))
            | (Column::Temporal { .. }, IdxKey::Temporal(..))
    )
}

/// Does the typed column hold a value at `row` satisfying `op key`?
///
/// Returns `None` when the column CANNOT answer — a missing key, an untyped
/// (`Mixed`) column, or an operand whose type differs from the column's. That
/// last one matters for more than speed: a cross-type comparison is a TYPE FAULT
/// in Gremlin and three-valued UNKNOWN in GQL, and neither is expressible as a
/// yes/no here. Declining lets the caller fall back to the path that gets those
/// right, so this filter is only ever used where the answer is unambiguous.
fn column_matches<'g>(
    graph: &'g Graph,
    key: &str,
    edge: bool,
    op: SeekOp,
    want: &IdxKey,
) -> Option<Box<dyn Fn(u32) -> bool + 'g>> {
    let store = if edge {
        &graph.edge_props
    } else {
        &graph.props
    };
    let col = store.cols.get(store.keys.get(key)? as usize)?;

    match (col, want) {
        (Column::Num { data, present }, IdxKey::Num(n)) => {
            let n = *n;

            Some(Box::new(move |row| {
                let i = row as usize;

                present.get(i)
                    && match op {
                        SeekOp::Eq => data[i] == n,
                        SeekOp::Lt => data[i] < n,
                        SeekOp::Le => data[i] <= n,
                        SeekOp::Gt => data[i] > n,
                        SeekOp::Ge => data[i] >= n,
                    }
            }))
        }
        // Strings compare by INTERNED ID, so only equality — an ordering would
        // need the dictionary text per row, which is the slow path this exists to
        // avoid. A value never interned matches nothing, which is correct.
        (Column::Str { data, present }, IdxKey::Str(s)) if op == SeekOp::Eq => {
            let want_id = graph.strs.get(s);

            Some(Box::new(move |row| {
                let i = row as usize;

                present.get(i) && want_id.is_some_and(|w| data[i] == w)
            }))
        }
        (Column::Bool { data, present }, IdxKey::Bool(b)) if op == SeekOp::Eq => {
            let b = *b;

            Some(Box::new(move |row| {
                let i = row as usize;

                present.get(i) && data[i] == b
            }))
        }
        _ => None,
    }
}

/// A folded bound as a seek. An exact point uses the POINT lookup rather than a
/// degenerate one-element range — `k = 'x'` is by far the most common predicate
/// there is, and routing it through the range machinery would make every
/// equality pay for a scan it does not need.
fn seek_bound(graph: &Graph, name: &str, rb: &RangeBound, edge: bool) -> Option<Vec<u32>> {
    match (&rb.gte, &rb.lte, &rb.gt, &rb.lt) {
        (Some(lo), Some(hi), None, None) if lo == hi => idx_eq(graph, name, lo, edge),
        _ => idx_range(graph, name, rb, edge),
    }
}

fn idx_eq(graph: &Graph, name: &str, k: &IdxKey, edge: bool) -> Option<Vec<u32>> {
    if edge {
        graph.edges_by_prop(name, k).map(<[u32]>::to_vec)
    } else {
        graph.vertices_by_prop(name, k).map(<[u32]>::to_vec)
    }
}

fn idx_range(graph: &Graph, name: &str, rb: &RangeBound, edge: bool) -> Option<Vec<u32>> {
    if edge {
        graph.edges_by_prop_range(name, rb)
    } else {
        graph.vertices_by_prop_range(name, rb)
    }
}

#[cfg(test)]
mod tests;
