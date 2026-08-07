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
use crate::value::Value;

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

/// A label constraint on one element variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelFilter {
    /// Vertex-label ids, or edge-type ids when the seek is over edges. An empty
    /// list matches nothing (the name resolved to no id).
    pub ids: Vec<u32>,
}

/// One branch of an `OR` — its own conjunction of predicates.
pub type Branch = Vec<KeyPredicate>;

/// One branch of a disjunction, compiled to column tests — every one must hold
/// for that branch to match.
type BranchTests<'g> = Vec<Box<dyn Fn(u32) -> bool + 'g>>;
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
    /// The label constraint, when the front end could express it as a flat id
    /// list. GQL's boolean label expressions (`!A`, `A&B`, wildcard) do not fit
    /// and keep their own evaluator — lowering what fits is the point, not
    /// lowering everything.
    labels: Option<LabelFilter>,
    /// Key PRESENCE, conjunctive like [`Self::conj`]: `(key, must_be_present)`.
    ///
    /// Every column already carries a `present` bitmap and every [`SeekOp`] tests
    /// it, so this is the one predicate the IR could not spell despite the
    /// storage supporting it directly. Gremlin's `has(k)` / `hasKey(k)` /
    /// `hasNot(k)` are exactly it.
    ///
    /// Presence is NOT "is not null" — a stored null is PRESENT in this engine
    /// (see the null-as-a-value model). GQL's `IS NOT NULL` is a value test and
    /// does not lower here; `IS NULL` over an absent key is true, which presence
    /// alone cannot answer.
    ///
    /// It never SEEDS. An index maps values to elements and absence has no value
    /// to look up, so this only ever narrows candidates a bucket or another
    /// predicate produced.
    presence: Vec<(Arc<str>, bool)>,
    /// NEGATED comparisons, conjunctive: each holds when the element does NOT
    /// satisfy it — `NOT (the key is present AND the comparison holds)`.
    ///
    /// That reading is the point, and it is why this is its own list rather than
    /// an inverted [`SeekOp`]. Gremlin's `not(has(k, v))` is satisfied by an
    /// element with no `k` AT ALL, so `k != v` as a comparison would drop exactly
    /// the rows the traversal asks for. A STORED NULL is the same case one step
    /// in: it is present and satisfies no comparison, so it satisfies every
    /// negation of one — where a `<` OR `>` disjunction standing in for a negated
    /// equality would drop it.
    ///
    /// Like [`Self::presence`] it never seeds: an index finds the elements that
    /// DO match a value, and there is no lookup for "everything else".
    negated: Vec<KeyPredicate>,
}

impl ElementSeek {
    #[must_use]
    pub fn node() -> Self {
        Self {
            edge: false,
            conj: Vec::new(),
            disj: Vec::new(),
            labels: None,
            presence: Vec::new(),
            negated: Vec::new(),
        }
    }

    #[must_use]
    pub fn edge() -> Self {
        Self {
            edge: true,
            conj: Vec::new(),
            disj: Vec::new(),
            labels: None,
            presence: Vec::new(),
            negated: Vec::new(),
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

    /// True when this constrains NOTHING about an element's properties — the
    /// caller should scan, and a shortcut that answers from a bucket alone is
    /// entitled to.
    ///
    /// A label filter may still be present. "Nothing at all" is
    /// `is_empty() && labels().is_none()` — reading this as the latter let a
    /// `hasLabel(…)` be silently ignored by a count shortcut.
    ///
    /// This counts PRESENCE and [`conj_is_empty`](Self::conj_is_empty) does not,
    /// which is the one place the two differ. A presence test narrows, so a
    /// bucket-length answer would be wrong; but it also never seeds, so it is not
    /// a reason to refuse to lower. The two questions are "may I skip the scan
    /// entirely" and "did anything here give me a seed", and a presence test
    /// answers them differently.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conj.is_empty()
            && self.disj.is_empty()
            && self.presence.is_empty()
            && self.negated.is_empty()
    }

    /// The label constraint, if one was lowered.
    #[must_use]
    pub fn labels(&self) -> Option<&LabelFilter> {
        self.labels.as_ref()
    }

    /// True when nothing here can produce a SEED (a label may still). Presence
    /// deliberately does not count — see [`is_empty`](Self::is_empty).
    #[must_use]
    pub fn conj_is_empty(&self) -> bool {
        self.conj.is_empty() && self.disj.is_empty()
    }

    /// Narrow `ids` to those satisfying every disjunction — each an OR of
    /// branches, each branch an AND of comparisons.
    ///
    /// A disjunction used to be a SEED and nothing else: `resolve` unions the
    /// branches' index lookups, and if there was no index to look up, the
    /// disjunction was silently not applied at all. That is why [`columnar`] has
    /// always refused to call a seek with one "equivalent to the front end's own
    /// filters" — it was not, and every caller that asked had to fall back to
    /// running the query the slow way.
    ///
    /// Applying it here makes the seek exact whether or not an index answered it,
    /// which is what lets the gate relax. Re-applying a disjunction that DID seed
    /// is redundant, not wrong: it selects the same elements a second time.
    ///
    /// A branch whose operand does not resolve, or whose comparison the column
    /// cannot run, makes that branch match NOTHING here — so the disjunction is
    /// only applied when [`columnar`] has established that every branch resolves.
    fn retain_disj(&self, graph: &Graph, param: &impl Bindings, ids: &mut Vec<u32>) {
        for d in &self.disj {
            if let Some(tests) = self.disj_tests(graph, param, d) {
                ids.retain(|&id| tests.iter().any(|b| b.iter().all(|t| t(id))));
            }
        }
    }

    /// One disjunction compiled to per-branch column tests, or `None` when it
    /// cannot be run column-at-a-time.
    ///
    /// A branch that lost a comparison on the way in would match TOO MUCH, so an
    /// incomplete lowering drops the whole disjunction rather than applying a
    /// widened version of it. Dropping is safe in both directions: the caller
    /// only treats the seek as exact when [`columnar`](Self::columnar) agreed,
    /// and that asks the same question this does.
    fn disj_tests<'g>(
        &self,
        graph: &'g Graph,
        param: &impl Bindings,
        d: &Disjunction,
    ) -> Option<Vec<BranchTests<'g>>> {
        let branches: Vec<Branch> = match d {
            Disjunction::Branches(bs) => bs.clone(),
            Disjunction::AnyOfParam { key, slot } => param
                .list(*slot)?
                .into_iter()
                .map(|v| {
                    vec![KeyPredicate {
                        key: key.clone(),
                        op: SeekOp::Eq,
                        operand: Operand::Lit(v),
                    }]
                })
                .collect(),
        };
        let tests: Vec<Vec<_>> = branches
            .iter()
            .map(|b| {
                b.iter()
                    .filter_map(|p| {
                        p.operand
                            .resolve(param)
                            .and_then(|k| column_matches(graph, &p.key, self.edge, p.op, &k))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        tests
            .iter()
            .zip(&branches)
            .all(|(t, b)| t.len() == b.len())
            .then_some(tests)
    }

    /// Narrow `ids` to those satisfying every NEGATED comparison — for each, the
    /// element must not carry the key, or must carry it and fail the test.
    ///
    /// A comparison that cannot run column-at-a-time is SKIPPED rather than
    /// inverted: a negation applied to a widened test excludes rows, which is the
    /// direction that loses answers. [`columnar`](Self::columnar) asks the same
    /// question before any caller treats this seek as exact.
    fn retain_negated(&self, graph: &Graph, param: &impl Bindings, ids: &mut Vec<u32>) {
        for p in &self.negated {
            let Some(test) = p
                .operand
                .resolve(param)
                .and_then(|k| column_matches(graph, &p.key, self.edge, p.op, &k))
            else {
                continue;
            };

            // `column_matches` already tests presence, so `!test(id)` is exactly
            // "absent, or present and not matching".
            ids.retain(|&id| !test(id));
        }
    }

    /// Narrow `ids` to those whose presence tests all hold.
    ///
    /// A key the store has never seen has no column, so nothing carries it:
    /// everything fails a `must be present` and everything passes a `must be
    /// absent`.
    fn retain_present(&self, graph: &Graph, ids: &mut Vec<u32>) {
        let store = if self.edge {
            &graph.edge_props
        } else {
            &graph.props
        };

        for (key, want) in &self.presence {
            // `is_present_id` rather than a `present` bitmap directly: a `Mixed`
            // column has none (presence is `Some`), and a `Record` column counts
            // an escaped value as present. Reading the bitmap would have been
            // right for the four packed shapes and wrong for the other two.
            match store.keys.get(key) {
                Some(kid) => ids.retain(|&i| store.is_present_id(i as usize, kid) == *want),
                // No column at all: nothing carries the key.
                None if *want => {
                    ids.clear();
                    return;
                }
                None => {}
            }
        }
    }

    /// Constrain to elements carrying one of `ids`, under `rule`.
    pub fn set_labels(&mut self, ids: Vec<u32>) {
        self.labels = Some(LabelFilter { ids });
    }

    /// Does this element satisfy the label constraint?
    #[must_use]
    fn label_ok(&self, graph: &Graph, id: u32) -> bool {
        let Some(f) = &self.labels else {
            return true;
        };

        if self.edge {
            // ANY of the edge's labels — edges are multi-label like vertices.
            return f.ids.iter().any(|&t| graph.edge_has_label(id, t));
        }

        // ANY of the element's labels. Both languages want this: ISO GQL's
        // `(n:Person)` by definition, and Gremlin's `hasLabel` because the TS
        // engine is `labels.some(...)`. Native once matched only the FIRST label,
        // which was a byte-identity divergence dressed up as a TinkerPop
        // contract — TinkerPop has one label per vertex and says nothing here.
        f.ids.iter().any(|&l| graph.has_label(id, l))
    }

    /// Candidates from the label buckets, when a label constraint is the only
    /// thing narrowing the set.
    ///
    /// Valid under BOTH rules, which is why it can live here: bucket membership
    /// is "carries this label anywhere", so it is exactly the `Any` answer and a
    /// SUPERSET of the `First` answer — and `label_ok` re-checks either way.
    ///
    /// Returns `None` when the buckets are not SMALLER than the whole live set,
    /// because then seeding from them is strictly more work than scanning: the
    /// first version of this always seeded and measured WORSE (1.22 -> 1.87 ms)
    /// on a fixture where one label covers 96% of the graph. A label is only a
    /// useful seed when it is selective, which is a decision the IR can make once
    /// for both engines rather than each guessing.
    /// Seed ids from the label buckets, or `None` if that is no narrower than the
    /// whole universe.
    ///
    /// `take` bounds how many are collected. A capped scan with nothing left to
    /// filter — `MATCH (n:Person) RETURN n.name LIMIT 100` — needs 100 ids, not
    /// the 50,000 in the bucket; materializing the bucket first made that query
    /// 2.7x slower than the hand-written path it replaced. `None` means collect
    /// them all, which is required whenever a later test can still reject rows.
    fn label_seed(&self, graph: &Graph, live: usize, take: Option<usize>) -> Option<Vec<u32>> {
        let f = self.labels.as_ref()?;

        if self.edge {
            let total: usize = f.ids.iter().map(|&t| graph.edges_with_etype(t).len()).sum();

            if total >= live {
                return None;
            }

            let want = take.unwrap_or(total).min(total);
            let mut out = Vec::with_capacity(want);
            // An edge is bucketed under EVERY label it carries, so two of the
            // wanted labels can name the SAME edge — `[:X|Y]` over an edge
            // labelled [X, Y] returned it twice. Exactly the case the vertex
            // branch below has always deduped; edges only started needing it when
            // they became multi-label. Skipped entirely when one label is asked
            // for, or when no edge carries a second.
            let dedup = f.ids.len() > 1 && graph.has_multi_label_edges();
            let mut seen: HashSet<u32> = if dedup {
                HashSet::with_capacity(want)
            } else {
                HashSet::new()
            };

            for &t in &f.ids {
                for &e in graph.edges_with_etype(t) {
                    if out.len() >= want {
                        break;
                    }

                    if !dedup || seen.insert(e) {
                        out.push(e);
                    }
                }
            }

            return Some(out);
        }

        let total: usize = f
            .ids
            .iter()
            .map(|&l| graph.vertices_with_label(l).len())
            .sum();

        // A bucket no smaller than the live set is not a SELECTIVE seed — but it
        // is still the cheaper one, because declining here hands the caller the
        // whole universe AND an obligation to re-check the label per vertex.
        // Copying the bucket is one memcpy; the re-check is a pointer chase per
        // row. Declining when `total >= live` cost 2.26x on
        // `MATCH (n:Person) RETURN sum(n.age)` over a fixture where every vertex
        // is a Person — invisible at 52k, where this was validated, and plain at
        // 1M. Only a MULTI-label ask still declines: there the dedup set is a
        // real cost the re-check may beat.
        if total >= live && f.ids.len() > 1 {
            return None;
        }

        let want = take.unwrap_or(total).min(total);

        // One label cannot produce a duplicate, so the common case skips the set.
        if let [only] = f.ids.as_slice() {
            let src = graph.vertices_with_label(*only);

            return Some(src[..want.min(src.len())].to_vec());
        }

        let mut out = Vec::with_capacity(want);
        let mut seen = HashSet::with_capacity(want);

        for &l in &f.ids {
            if out.len() >= want {
                break;
            }

            // A vertex is bucketed under EVERY label it carries, so two labels
            // can name the same vertex.
            for &v in graph.vertices_with_label(l) {
                if out.len() >= want {
                    break;
                }

                if seen.insert(v) {
                    out.push(v);
                }
            }
        }

        Some(out)
    }

    /// A `key OP value` conjunct.
    pub fn push(&mut self, key: Arc<str>, op: SeekOp, operand: Operand) {
        self.conj_push(KeyPredicate { key, op, operand });
    }

    /// Require `key` to be present (or, with `want` false, absent).
    pub fn push_presence(&mut self, key: Arc<str>, want: bool) {
        self.presence.push((key, want));
    }

    /// Require that `p` does NOT hold — including for an element that does not
    /// carry the key at all. See [`Self::negated`].
    pub fn push_negated(&mut self, p: KeyPredicate) {
        self.negated.push(p);
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

    /// How many elements satisfy this seek, without materializing any of them.
    ///
    /// `count()` is the shape where the difference shows most: answering it by
    /// building one traverser per element and taking the length was 1.2 ms where
    /// the count itself is a walk over a bucket.
    #[must_use]
    pub fn count(
        &self,
        graph: &Graph,
        param: &impl Bindings,
        universe: impl FnOnce() -> Vec<u32>,
    ) -> usize {
        self.scan(graph, param, universe).len()
    }

    /// Whether every conjunct can be answered straight from a typed column.
    ///
    /// Only then is [`scan`](Self::scan) equivalent to running the front end's
    /// own filters — see [`column_matches`] for why the types must line up.
    #[must_use]
    pub fn columnar(&self, graph: &Graph, param: &impl Bindings) -> bool {
        let runs = |p: &KeyPredicate| {
            p.operand
                .resolve(param)
                .and_then(|k| column_matches(graph, &p.key, self.edge, p.op, &k))
                .is_some()
        };

        (!self.conj.is_empty() || !self.disj.is_empty() || !self.negated.is_empty())
            && self.conj.iter().all(runs)
            // A negation is only exact if the thing being negated can run.
            && self.negated.iter().all(runs)
            // A disjunction used to disqualify a seek outright, because it was a
            // SEED and nothing more — with no index it simply did not apply. Now
            // `retain_disj` runs it column-at-a-time like the conjuncts, so the
            // question is the same one: can every comparison in it run.
            //
            // `AnyOfParam` holds values that do not exist until the parameters are
            // bound, so the branches cannot be checked here. It keeps the old
            // answer.
            && self.disj.iter().all(|d| match d {
                Disjunction::Branches(bs) => bs.iter().all(|b| b.iter().all(runs)),
                Disjunction::AnyOfParam { .. } => false,
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
    /// Call only when [`columnar`](Self::columnar) holds — or when the only
    /// constraint is a label, which is always answerable.
    #[must_use]
    pub fn scan(
        &self,
        graph: &Graph,
        param: &impl Bindings,
        universe: impl FnOnce() -> Vec<u32>,
    ) -> Vec<u32> {
        self.scan_capped(graph, param, None, universe)
    }

    /// As [`scan`](Self::scan), stopping once `cap` rows survive every filter.
    ///
    /// The cap has to be pushed INTO the walk, not applied after it: truncating
    /// a materialized result made `RETURN … LIMIT 100` over a large label 92x
    /// slower, because the work is the scan, not the truncation.
    #[must_use]
    pub fn scan_capped(
        &self,
        graph: &Graph,
        param: &impl Bindings,
        cap: Option<usize>,
        universe: impl FnOnce() -> Vec<u32>,
    ) -> Vec<u32> {
        // The residual here is constant-true, so nothing after the seed can
        // reject a row and the seed itself may stop at `cap`. That is ONLY true
        // on this path — `scan_with`'s caller-supplied residual can reject, so it
        // passes `None` and takes the whole bucket.
        self.scan_inner(graph, param, cap, cap, universe, |_| true)
    }

    /// As [`scan_capped`](Self::scan_capped), with a RESIDUAL predicate the
    /// front end supplies for whatever the IR could not express.
    ///
    /// This is what lets one scan loop serve both engines. The seed, the cap, the
    /// label rule and the column filters live here; the part that is genuinely
    /// per-language — GQL's arbitrary `WHERE` over a binding, Gremlin's
    /// predicates that are not a range — arrives as a closure. Neither engine
    /// needs its own loop to get its own semantics.
    ///
    /// The residual runs LAST, after the cheap tests have already rejected most
    /// candidates, and only until `cap` rows survive.
    #[must_use]
    pub fn scan_with(
        &self,
        graph: &Graph,
        param: &impl Bindings,
        cap: Option<usize>,
        universe: impl FnOnce() -> Vec<u32>,
        residual: impl FnMut(u32) -> bool,
    ) -> Vec<u32> {
        // `seed_take` is None: the residual can reject a candidate, so a seed cut
        // short at `cap` could return fewer rows than exist.
        self.scan_inner(graph, param, cap, None, universe, residual)
    }

    fn scan_inner(
        &self,
        graph: &Graph,
        param: &impl Bindings,
        cap: Option<usize>,
        seed_cap: Option<usize>,
        universe: impl FnOnce() -> Vec<u32>,
        mut residual: impl FnMut(u32) -> bool,
    ) -> Vec<u32> {
        let live = if self.edge {
            graph.edge_count()
        } else {
            graph.vertex_count()
        };
        // Seeding from the buckets under the ANY rule already proves the label:
        // a vertex is bucketed under every label it carries, so union membership
        // IS "carries one of these". Re-checking is pure waste, and skipping it
        // is what the hand-written GQL bucket path did — losing it cost 50% on a
        // grouped aggregate. Under FIRST the buckets are only a superset, so the
        // check stays.
        // Bounding the SEED is only sound when nothing after it can reject a row.
        // The caller establishes that there is no residual (`seed_cap` is only set
        // by `scan_capped`); the remaining conditions are local: no conjunct left
        // to test, and a bucket that already PROVES the label rather than merely
        // over-approximating it.
        // A bucket seed already PROVES the label — a vertex is bucketed under
        // every label it carries, so union membership IS "carries one of these".
        // Nothing after it can reject on the label, which is both why the seed
        // may stop at the cap and why the re-check below is skipped.
        let seed_take = seed_cap.filter(|_| self.conj.is_empty() && self.labels.is_some());
        let mut label_checked = false;
        let mut ids = match self.resolve(graph, param) {
            Some(seeded) => seeded,
            None => match self.label_seed(graph, live, seed_take) {
                Some(bucketed) => {
                    label_checked = true;
                    bucketed
                }
                None => universe(),
            },
        };

        let labelled = self.labels.is_some() && !label_checked;

        // Uncapped is the hot shape: narrow column-at-a-time in monomorphic
        // loops, then run the residual over what survives. A capped scan takes
        // the single-pass form below instead, because it has to stop early.
        if cap.is_none() {
            if labelled {
                ids.retain(|&id| self.label_ok(graph, id));
            }

            for p in &self.conj {
                if let Some(k) = p.operand.resolve(param) {
                    retain_matching(graph, &mut ids, &p.key, self.edge, p.op, &k);
                }
            }

            self.retain_present(graph, &mut ids);
            self.retain_negated(graph, param, &mut ids);
            self.retain_disj(graph, param, &mut ids);
            ids.retain(|&id| residual(id));

            return ids;
        }

        // Capped: one pass, cheapest test first, stopping as soon as enough rows
        // survive. The boxed tests cost a virtual call each, which is the right
        // trade only because a cap keeps the row count small.
        let tests: Vec<_> = self
            .conj
            .iter()
            .filter_map(|p| {
                p.operand
                    .resolve(param)
                    .and_then(|k| column_matches(graph, &p.key, self.edge, p.op, &k))
            })
            .collect();
        // Presence resolves to a key id once and then tests per element, like the
        // conjuncts above it. A key with no column fails every `present` test, so
        // an unresolvable one becomes a test that always says no rather than a
        // test that is skipped.
        let store = if self.edge {
            &graph.edge_props
        } else {
            &graph.props
        };
        let presence: Vec<(Option<u32>, bool)> = self
            .presence
            .iter()
            .map(|(k, want)| (store.keys.get(k), *want))
            .collect();
        // The disjunctions, in the same per-element form. Leaving them out of this
        // path was invisible while a disjunction disqualified the seek outright;
        // now that `columnar` accepts one, a capped scan that ignored it would
        // return the wrong rows rather than merely too many.
        let disj: Vec<_> = self
            .disj
            .iter()
            .filter_map(|d| self.disj_tests(graph, param, d))
            .collect();
        let negated: Vec<_> = self
            .negated
            .iter()
            .filter_map(|p| {
                p.operand
                    .resolve(param)
                    .and_then(|k| column_matches(graph, &p.key, self.edge, p.op, &k))
            })
            .collect();
        let c = cap.unwrap_or(usize::MAX);
        let mut out = Vec::with_capacity(c.min(ids.len()));

        for id in ids.drain(..) {
            if (!labelled || self.label_ok(graph, id))
                && tests.iter().all(|t| t(id))
                && negated.iter().all(|t| !t(id))
                && disj
                    .iter()
                    .all(|d| d.iter().any(|b| b.iter().all(|t| t(id))))
                && presence.iter().all(|(kid, want)| {
                    kid.is_some_and(|k| store.is_present_id(id as usize, k)) == *want
                })
                && residual(id)
            {
                out.push(id);

                if out.len() == c {
                    break;
                }
            }
        }

        out
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

/// Drop every id whose column value fails `op want`, in a MONOMORPHIC loop.
///
/// `column_matches` returns a boxed closure, which costs a virtual call per
/// element — fine for a capped scan of a few rows, 22% on a full one. Matching
/// the column and the operator ONCE and then running a tight typed loop is what
/// the hand-written GQL path did, and it is why routing through the shared layer
/// has to do the same.
///
/// Returns `false` when the column cannot answer, leaving `ids` untouched so the
/// caller can fall back.
fn retain_matching(
    graph: &Graph,
    ids: &mut Vec<u32>,
    key: &str,
    edge: bool,
    op: SeekOp,
    want: &IdxKey,
) -> bool {
    let store = if edge {
        &graph.edge_props
    } else {
        &graph.props
    };
    let Some(col) = store.keys.get(key).and_then(|k| store.cols.get(k as usize)) else {
        return false;
    };

    macro_rules! by_op {
        ($data:expr, $present:expr, $v:expr) => {{
            let (data, present, v) = ($data, $present, $v);

            match op {
                SeekOp::Eq => ids.retain(|&i| present.get(i as usize) && data[i as usize] == v),
                SeekOp::Lt => ids.retain(|&i| present.get(i as usize) && data[i as usize] < v),
                SeekOp::Le => ids.retain(|&i| present.get(i as usize) && data[i as usize] <= v),
                SeekOp::Gt => ids.retain(|&i| present.get(i as usize) && data[i as usize] > v),
                SeekOp::Ge => ids.retain(|&i| present.get(i as usize) && data[i as usize] >= v),
            }
        }};
    }

    match (col, want) {
        (Column::Num { data, present }, IdxKey::Num(n)) => by_op!(data, present, *n),
        // Strings compare by INTERNED ID, so equality only — an ordering would
        // need the dictionary text per row. A value never interned matches
        // nothing, which is correct.
        (Column::Str { data, present }, IdxKey::Str(t)) if op == SeekOp::Eq => {
            match graph.strs.get(t) {
                Some(w) => ids.retain(|&i| present.get(i as usize) && data[i as usize] == w),
                None => ids.clear(),
            }
        }
        (Column::Bool { data, present }, IdxKey::Bool(b)) if op == SeekOp::Eq => {
            let b = *b;

            ids.retain(|&i| present.get(i as usize) && data[i as usize] == b);
        }
        _ => return false,
    }

    true
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

/// Which way an expansion walks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    Out,
    In,
    Both,
}

/// How an UNDIRECTED expansion treats a self-loop.
///
/// A self-loop sits in both the out- and the in-index of its vertex, so walking
/// `Both` reaches it from each side. The two languages disagree about whether
/// that is one traversal or two, and both are right for their own model:
///
/// - GQL's `(a)-[:R]-(b)` matches the loop ONCE — it is one edge, matched once.
/// - TinkerPop's `both()` yields it TWICE — one traverser per direction.
///
/// So the rule is data on the IR node, like [`LabelRule`]. Getting this wrong is
/// silent: the row count is simply off by the number of self-loops.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SelfLoops {
    /// Emit a self-loop once from the out-side (GQL).
    Once,
    /// Emit it from both sides (TinkerPop).
    Twice,
}

/// Does adjacency entry `a` match `etypes` (empty = any type)?
///
/// The ONE edge-label rule, shared by the per-vertex iterator and the bulk
/// expansions. `need_extra` is [`Graph::etypes_need_extra_lookup`], hoisted by
/// the caller: an edge's FIRST label is mirrored here, so only a query for a
/// label that some edge carries as a SECOND one can need the sparse lookup.
///
/// A free function rather than a method on the iterator because the bulk
/// expansions cannot afford `adj`'s chained-iterator form — routing them through
/// it measured ~1.4x on a plain `out()` count, the same way it did for GQL's
/// count shortcuts.
#[inline]
pub fn adj_keeps(graph: &Graph, a: &crate::graph::Adj, etypes: &[u32], need_extra: bool) -> bool {
    etypes.is_empty()
        || etypes.contains(&a.etype)
        || (need_extra && etypes.iter().any(|&t| graph.edge_has_label(a.eidx, t)))
}

/// Every incident edge of `v` whose type is in `etypes` (empty = any), in
/// ADJACENCY order: out-edges then in-edges, each in storage order.
///
/// The one adjacency walk both engines use. Gremlin previously had a second,
/// `adj_in_label_order`, which materialized a `Vec` per direction per source
/// vertex and re-scanned it once per label argument so that
/// `out('a','b')` emitted all the `a` edges before all the `b` edges. That order
/// is NOT contractual — `docs/` records adjacency order as unspecified, and the
/// native and TS engines already disagree on it (CSR order vs label buckets) —
/// so paying for it was buying nothing.
pub fn adj<'a>(
    graph: &'a Graph,
    v: u32,
    dir: Dir,
    etypes: &'a [u32],
    loops: SelfLoops,
) -> impl Iterator<Item = crate::graph::Adj> + 'a {
    // An edge's FIRST label is mirrored into its adjacency entry, so the common
    // filter is one `u32` compare. A miss falls through to the edge's other
    // labels ONLY when one of the labels asked for is actually carried as a
    // secondary somewhere — decided once here, not per edge. Testing merely
    // "does the graph hold any multi-label edge?" cost 2.4x on every traversal
    // as soon as a single edge gained a second label.
    let need_extra = graph.etypes_need_extra_lookup(etypes);

    adj_where(graph, v, dir, loops, move |a| {
        adj_keeps(graph, a, etypes, need_extra)
    })
}

/// [`adj`] with an arbitrary predicate instead of a type set.
///
/// The WALK is here and the FILTER is the caller's: which indexes to read for a
/// direction, in what order, and the self-loop rule — a loop sits in both the
/// out- and in-index of its vertex, so an undirected walk that read both would
/// yield it twice.
///
/// That rule is the reason this exists as its own function. GQL had a second copy
/// of the whole walk, self-loop comment and all, for one reason: its label filter
/// is a boolean EXPRESSION (`!A`, `A&B`, wildcards) rather than a flat type set,
/// so it could not call `adj`. That is a difference in the predicate, and a
/// predicate is a parameter.
pub fn adj_where<'a>(
    graph: &'a Graph,
    v: u32,
    dir: Dir,
    loops: SelfLoops,
    keep: impl Fn(&crate::graph::Adj) -> bool + 'a,
) -> impl Iterator<Item = crate::graph::Adj> + 'a {
    // Only an undirected walk can reach a self-loop twice; a directed one keeps
    // it either way.
    let drop_loop = dir == Dir::Both && loops == SelfLoops::Once;
    let outs = (dir != Dir::In)
        .then(|| graph.out_adj(v))
        .into_iter()
        .flatten();
    let ins = (dir != Dir::Out)
        .then(|| graph.in_adj(v))
        .into_iter()
        .flatten()
        .filter(move |a| !(drop_loop && a.nbr == v));

    outs.chain(ins).filter(move |a| keep(a))
}

/// Neighbours of `src` along `etypes` (empty = any type), flat.
///
/// One output vector for the whole expansion rather than one per source. The
/// per-traverser walk allocated a `Vec` for EACH source vertex — 50k allocations
/// to expand a 50k-vertex set — and then a traverser per neighbour on top.
///
/// Duplicates are kept: two edges between the same pair are two traversers in
/// Gremlin and two rows in GQL, so de-duplicating here would silently change the
/// answer.
#[must_use]
pub fn expand(graph: &Graph, src: &[u32], dir: Dir, etypes: &[u32], loops: SelfLoops) -> Vec<u32> {
    let mut out = Vec::new();
    let need_extra = graph.etypes_need_extra_lookup(etypes);
    let keep = |a: &crate::graph::Adj| adj_keeps(graph, a, etypes, need_extra);
    // Only an undirected walk can reach a loop twice; a directed one keeps it
    // either way.
    let drop_loop = dir == Dir::Both && loops == SelfLoops::Once;

    for &v in src {
        if dir != Dir::In {
            out.extend(graph.out_adj(v).filter(|a| keep(a)).map(|a| a.nbr));
        }

        if dir != Dir::Out {
            out.extend(
                graph
                    .in_adj(v)
                    .filter(|a| keep(a) && !(drop_loop && a.nbr == v))
                    .map(|a| a.nbr),
            );
        }
    }

    out
}

/// The EDGES incident to `src` along `etypes` (empty = any), flat.
///
/// `outE`/`inE`/`bothE` land on the edge itself rather than its far end, so the
/// result is an edge-id frontier. Self-loops follow the same rule as [`expand`]:
/// an undirected walk reaches one from both sides.
#[must_use]
pub fn expand_edges(
    graph: &Graph,
    src: &[u32],
    dir: Dir,
    etypes: &[u32],
    loops: SelfLoops,
) -> Vec<u32> {
    let mut out = Vec::new();
    let need_extra = graph.etypes_need_extra_lookup(etypes);
    let keep = |a: &crate::graph::Adj| adj_keeps(graph, a, etypes, need_extra);
    let drop_loop = dir == Dir::Both && loops == SelfLoops::Once;

    for &v in src {
        if dir != Dir::In {
            out.extend(graph.out_adj(v).filter(|a| keep(a)).map(|a| a.eidx));
        }

        if dir != Dir::Out {
            out.extend(
                graph
                    .in_adj(v)
                    .filter(|a| keep(a) && !(drop_loop && a.nbr == v))
                    .map(|a| a.eidx),
            );
        }
    }

    out
}

/// Walk `seed` through `hops`, borrowing until something actually expands.
///
/// THE streaming walk: a frontier in, a frontier out, no frame and no pairing.
/// Both engines had their own copy of this loop — `walk_count`'s prefix,
/// `streamed_frame`'s, and Gremlin's `lowered_ids` — differing only in what they
/// did with the result.
///
/// `hops` follows [`Hop::etypes`]: `None` is ANY type, `Some(&[])` is NONE, and
/// a hop that matches nothing makes the whole walk empty.
fn walk<'a>(
    graph: &Graph,
    seed: &'a [u32],
    hops: &[(Dir, Option<Vec<u32>>)],
    loops: SelfLoops,
) -> std::borrow::Cow<'a, [u32]> {
    if hops
        .iter()
        .any(|(_, e)| e.as_ref().is_some_and(Vec::is_empty))
    {
        return std::borrow::Cow::Owned(Vec::new());
    }

    // Borrowed until something expands, so a walk of no hops — and the LAST hop
    // of a count, which is counted rather than built — never copies the seed.
    let mut cur: std::borrow::Cow<'_, [u32]> = std::borrow::Cow::Borrowed(seed);
    // `expand` reads an empty list as ANY, which is what `None` means here.
    let any: &[u32] = &[];

    for (d, e) in hops {
        cur = std::borrow::Cow::Owned(expand(graph, &cur, *d, e.as_deref().unwrap_or(any), loops));
    }

    cur
}

/// The frontier a walk lands on.
///
/// The STREAM route for a terminal that wants the ROWS rather than a fold — the
/// sibling of [`walk_count`], and what a per-language expansion loop was doing.
#[must_use]
pub fn walk_ids(
    graph: &Graph,
    seed: &[u32],
    hops: &[(Dir, Option<Vec<u32>>)],
    loops: SelfLoops,
) -> Vec<u32> {
    walk(graph, seed, hops, loops).into_owned()
}

/// The STREAM route for a reducing terminal: count a walk without building it.
///
/// Walk every hop but the last, then count the last in place. The rows are never
/// materialized because nothing needs them all at once — which is what
/// "reducing" means — consume every row, emit one, hold no buffer — and the
/// reason a frame is pure
/// cost here.
///
/// Shared because it is one operation in two languages. Gremlin reached it as
/// `g.V().hasLabel('V').has('n', gt(900)).out('R').count()` and GQL wrote the
/// same question as `MATCH (a:V)-[:R]->(b) WHERE a.n > 900 RETURN count(*)` and
/// built a frame of 14,850 pairs to count them: the hop cost 0.026ms one way and
/// 0.144ms the other, 5.5x, for identical work on identical storage.
///
/// `distinct` materializes the LAST hop, because deduplicating means holding the
/// endpoints — a `DISTINCT` cannot emit until every row exists, and this
/// is the one place the two routes meet. Everything before it still streams.
///
/// `hops` is `(direction, edge types)` per segment, under [`Hop::etypes`]'s
/// convention: `None` is ANY type, `Some(&[])` is NONE — a name that resolved to
/// nothing. Those two must not be collapsed, and taking a bare `Vec` here did
/// collapse them: GQL's `lower_labels` returns an empty list for an unresolved
/// name, `expand` reads an empty list as "any", and `MATCH (a)-[:NONEXISTENT]->(b)`
/// counted every edge in the graph. That is the fifth time this exact conflation
/// has been written here, which is why the signature now refuses it.
///
/// (Gremlin's own `Hop` alias spells the same distinction the other way round —
/// `None` for nothing, `Some(vec![])` for any — so its call site maps between
/// them explicitly.)
///
/// The vertices from which a chain of hops reaches `far` — a semi-join, walked
/// BACKWARDS.
///
/// `EXISTS { (a)-[:T]->()-[:T]->(b) … }` and `where(__.out('T').out('T'))` are
/// one question, and neither asks WHICH walk — only whether one exists. Run
/// forward per row it is `O(rows · degree^hops)`: bounded by finding a single
/// walk, but that bound is the whole tree exactly when no walk exists, which is
/// the rows the caller is about to discard. Backwards it is
/// `O(degree · |level|)` per hop and visits each vertex once per level.
///
/// `far` is the far end the caller already narrowed — the language's own filter
/// on the last node, which is the part that is NOT shared. Everything after it
/// is: both engines walk the same adjacency the same way.
///
/// Returns a per-vertex-slot bitmap; index it with a vertex id.
#[must_use]
pub fn reach_back(
    graph: &Graph,
    hops: &[(Dir, Option<Vec<u32>>)],
    far: Vec<bool>,
    loops: SelfLoops,
) -> Vec<bool> {
    let mut level = far;

    for (dir, etypes) in hops.iter().rev() {
        // THIS module's convention, the one `walk` and `walk_count` use: `None`
        // is ANY type and `Some(&[])` is NO type. Gremlin's `resolve_etypes` is
        // the INVERSE of it, so its caller converts — getting that backwards
        // turns "matches nothing" into "matches everything", which is not a
        // subtle wrong answer.
        if etypes.as_ref().is_some_and(Vec::is_empty) {
            return vec![false; level.len()];
        }

        let etypes = etypes.as_deref().unwrap_or(&[]);
        let mut prev = vec![false; level.len()];

        for (v, reached) in level.iter().enumerate() {
            if !reached {
                continue;
            }

            // The hop is `x -dir-> v`, so walk `v` the other way to find `x`.
            let back = match dir {
                Dir::Out => Dir::In,
                Dir::In => Dir::Out,
                Dir::Both => Dir::Both,
            };

            for a in adj(graph, v as u32, back, etypes, loops) {
                prev[a.nbr as usize] = true;
            }
        }

        level = prev;
    }

    level
}

/// A hop with a filter on its node or edge cannot come here — the rows have to
/// exist to be filtered — so the caller lowers only bare hops.
#[must_use]
pub fn walk_count(
    graph: &Graph,
    seed: &[u32],
    hops: &[(Dir, Option<Vec<u32>>)],
    loops: SelfLoops,
    distinct: bool,
) -> usize {
    // A hop whose type name resolved to nothing matches nothing, so the whole
    // walk does. `walk` checks this for the hops it takes; the LAST one is not
    // one of them, so it is checked here.
    if hops
        .iter()
        .any(|(_, e)| e.as_ref().is_some_and(Vec::is_empty))
    {
        return 0;
    }

    let Some(((dir, etypes), init)) = hops.split_last() else {
        // No expansion: the seeded set itself. Still subject to `distinct`, since
        // a seed can repeat an id.
        return if distinct {
            distinct_len(seed)
        } else {
            seed.len()
        };
    };

    // Every hop but the last, through the shared walk; the last is COUNTED.
    let cur = walk(graph, seed, init, loops);
    let last = etypes.as_deref().unwrap_or(&[]);

    if distinct {
        return distinct_len(&expand(graph, &cur, *dir, last, loops));
    }

    expand_count(graph, &cur, *dir, last, loops)
}

/// Distinct count over dense ids. Element identity IS the index, so this needs no
/// key projection — the reason a `DISTINCT` over elements is cheap where a
/// `DISTINCT` over values is not.
fn distinct_len(ids: &[u32]) -> usize {
    let mut seen = crate::fxhash::FxHashSet::default();

    ids.iter().filter(|&&id| seen.insert(id)).count()
}

/// How many neighbours [`expand`] would produce, without producing them.
///
/// The counting form allocates nothing at all — the whole expansion becomes a
/// walk over the adjacency slices.
#[must_use]
pub fn expand_count(
    graph: &Graph,
    src: &[u32],
    dir: Dir,
    etypes: &[u32],
    loops: SelfLoops,
) -> usize {
    let need_extra = graph.etypes_need_extra_lookup(etypes);
    let keep = |a: &crate::graph::Adj| adj_keeps(graph, a, etypes, need_extra);
    let drop_loop = dir == Dir::Both && loops == SelfLoops::Once;
    let mut n = 0;

    for &v in src {
        if dir != Dir::In {
            n += graph.out_adj(v).filter(|a| keep(a)).count();
        }

        if dir != Dir::Out {
            n += graph
                .in_adj(v)
                .filter(|a| keep(a) && !(drop_loop && a.nbr == v))
                .count();
        }
    }

    n
}

/// One expansion step: which way, along which edge types, binding which slots.
pub struct Hop<'a> {
    pub dir: Dir,
    /// `None` means ANY edge type. `Some(&[])` means NONE — a type name that
    /// resolved to nothing. Collapsing those two turns a typo into a full
    /// expansion, which is how `[:NONEXISTENT]` started matching every edge.
    pub etypes: Option<&'a [u32]>,
    pub loops: SelfLoops,
    /// Slots the traversed edge and the reached node bind, if named.
    pub rel_slot: Option<usize>,
    pub node_slot: Option<usize>,
    /// Slots this hop RE-binds — already carrying a value from an earlier hop.
    ///
    /// A pattern may name the same variable twice (`(a)-[:R]->(b)-[:S]->(a)`), and
    /// the second occurrence is not a new binding but an equality: the reached
    /// element must be the one already bound. Without this the expansion has to
    /// refuse the pattern entirely.
    pub rejoin_rel: bool,
    pub rejoin_node: bool,
}

/// What a front end checks per candidate that the IR could not express.
///
/// Two methods rather than one closure so the per-ROW work stays per row: GQL
/// binds every already-known slot once before walking a vertex's neighbours, and
/// folding that into the per-neighbour check would repeat it for the whole
/// degree.
pub trait RowFilter {
    /// Called once per source row, before its neighbours. `cols` is the frontier
    /// as it stands, so prior slots can be read at `row`.
    fn row(&mut self, cols: &[Option<Vec<u32>>], row: usize) {
        let _ = (cols, row);
    }

    /// Keep this `(edge, neighbour)`?
    fn keep(&mut self, eidx: u32, nbr: u32) -> bool;
}

/// The frontier of a multi-hop expansion: one id column per bound slot, plus the
/// element each row currently sits on.
///
/// This is the shape both engines were maintaining separately. GQL carried a
/// column per bound pattern variable and replicated them as rows fanned out;
/// Gremlin's lowered prefix carried a single flat id list, which is the same
/// structure with one column. Sharing it means the fan-out, the column
/// replication, the LIMIT stop and the intermediate-size ceiling are written
/// once.
pub struct Frontier {
    cols: Vec<Option<Vec<u32>>>,
    /// Value columns, for slots whose per-row binding is not a single element.
    /// A GQL group variable is the list of one repetition's values, and a Gremlin
    /// `select(Pop.all, 'x')` after a `repeat` is the same list — the frontier has
    /// to fan these out exactly like the id columns, or a later hop leaves them
    /// short and every row past the first reads off the end.
    vals: Vec<Option<Vec<Value>>>,
    endpoint: Vec<u32>,
}

/// The frontier grew past the configured ceiling — see [`Frontier::expand`].
#[derive(Debug)]
pub struct TooWide;

impl Frontier {
    /// Start from `endpoint`, binding it to `slot` if the start is named.
    #[must_use]
    pub fn seed(endpoint: Vec<u32>, slot: Option<usize>, width: usize) -> Self {
        let mut cols: Vec<Option<Vec<u32>>> = (0..width.max(1)).map(|_| None).collect();

        if let Some(s) = slot {
            cols[s] = Some(endpoint.clone());
        }

        let vals = (0..width.max(1)).map(|_| None).collect();

        Self {
            cols,
            vals,
            endpoint,
        }
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.endpoint.len()
    }

    #[must_use]
    pub fn endpoint(&self) -> &[u32] {
        &self.endpoint
    }

    /// Take a bound column out of the frontier.
    pub fn take_column(&mut self, slot: usize) -> Option<Vec<u32>> {
        self.cols.get_mut(slot).and_then(Option::take)
    }

    /// Install an id column, aligned with the CURRENT rows — a slot bound before
    /// this frontier existed (a `WITH` that carried an element forward). Later
    /// hops fan it out like any other bound column.
    pub fn set_column(&mut self, slot: usize, ids: Vec<u32>) {
        if slot < self.cols.len() {
            self.cols[slot] = Some(ids);
        }
    }

    /// Install a value column, aligned with the CURRENT rows. Later hops fan it
    /// out with the id columns.
    pub fn set_values(&mut self, slot: usize, vals: Vec<Value>) {
        if slot < self.vals.len() {
            self.vals[slot] = Some(vals);
        }
    }

    /// Take a value column back out.
    pub fn take_values(&mut self, slot: usize) -> Option<Vec<Value>> {
        self.vals.get_mut(slot).and_then(Option::take)
    }

    /// Mark a slot as carrying values, so later hops replicate it.
    pub fn bind(&mut self, slot: usize) {
        if self.cols[slot].is_none() {
            self.cols[slot] = Some(Vec::new());
        }
    }

    /// Fan rows out to endpoints computed ELSEWHERE.
    ///
    /// `rows[k]` is the source row that produced `ends[k]`. Used where the
    /// reachable set is not one adjacency step — a var-length walk, whose bounds
    /// and repeated-element restriction live in the front end's own walker rather
    /// than being re-derived here.
    pub fn replicate(&mut self, rows: &[usize], ends: &[u32], slot: Option<usize>) {
        let width = self.cols.len();
        let mut new_cols: Vec<Option<Vec<u32>>> = (0..width)
            .map(|s| {
                (self.cols[s].is_some() || slot == Some(s)).then(|| Vec::with_capacity(rows.len()))
            })
            .collect();

        let mut new_vals: Vec<Option<Vec<Value>>> = (0..width)
            .map(|s| {
                self.vals[s]
                    .as_ref()
                    .map(|_| Vec::with_capacity(rows.len()))
            })
            .collect();

        for (k, &i) in rows.iter().enumerate() {
            for (s, col) in new_cols.iter_mut().enumerate() {
                let Some(col) = col else { continue };
                let v = if slot == Some(s) {
                    ends[k]
                } else if let Some(prior) = &self.cols[s] {
                    prior[i]
                } else {
                    continue;
                };

                col.push(v);
            }

            for (s, col) in new_vals.iter_mut().enumerate() {
                let (Some(col), Some(prior)) = (col, &self.vals[s]) else {
                    continue;
                };

                col.push(prior[i].clone());
            }
        }

        self.cols = new_cols;
        self.vals = new_vals;
        self.endpoint = ends.to_vec();
    }

    /// Fan every row out along `hop`, replicating bound columns.
    ///
    /// `cap` stops the build as soon as enough rows exist — only sound on the
    /// LAST hop, where no later filter can drop them. `budget` bounds the
    /// intermediate result: the cross-product of partial matches reaches billions
    /// on a dense graph, and it is checked INSIDE the build so a single runaway
    /// layer caps rather than materializing first.
    pub fn expand<F: RowFilter + ?Sized>(
        &mut self,
        graph: &Graph,
        hop: &Hop<'_>,
        budget: u64,
        cap: Option<usize>,
        f: &mut F,
    ) -> Result<(), TooWide> {
        let width = self.cols.len();
        let mut new_cols: Vec<Option<Vec<u32>>> = (0..width)
            .map(|s| self.cols[s].as_ref().map(|_| Vec::new()))
            .collect();

        if let Some(s) = hop.rel_slot {
            new_cols[s] = Some(Vec::new());
        }

        if let Some(s) = hop.node_slot {
            new_cols[s] = Some(Vec::new());
        }

        let mut new_vals: Vec<Option<Vec<Value>>> = (0..width)
            .map(|s| self.vals[s].as_ref().map(|_| Vec::new()))
            .collect();
        let mut new_endpoint: Vec<u32> = Vec::new();
        // The edge-label rule, once — `hop.etypes` is `None` for ANY type and
        // `Some(&[])` for none. Testing `a.etype` alone is only an edge's FIRST
        // label, which silently dropped every multi-label edge whose match was on
        // a later one.
        // `adj_keeps` reads an EMPTY type list as "any type"; here `Some(&[])`
        // means NONE — a name that resolved to nothing. Delegating the empty case
        // made `[:NONEXISTENT]` match every edge, which is the fourth time that
        // exact conflation has been written in this codebase.
        let need_extra = graph.etypes_need_extra_lookup(hop.etypes.unwrap_or(&[]));
        let keep_type = |a: &crate::graph::Adj| match hop.etypes {
            None => true,
            Some([]) => false,
            Some(ids) => adj_keeps(graph, a, ids, need_extra),
        };
        let drop_loop = hop.dir == Dir::Both && hop.loops == SelfLoops::Once;

        'rows: for i in 0..self.endpoint.len() {
            let v = self.endpoint[i];

            f.row(&self.cols, i);

            let out = (hop.dir != Dir::In)
                .then(|| graph.out_adj(v))
                .into_iter()
                .flatten();
            let inn = (hop.dir != Dir::Out)
                .then(|| graph.in_adj(v))
                .into_iter()
                .flatten()
                .filter(|a| !(drop_loop && a.nbr == v));

            for a in out.chain(inn).filter(|a| keep_type(a)) {
                // A re-bound slot is an EQUALITY, not a new binding: the element
                // reached must be the one the earlier hop already put there.
                if hop.rejoin_node {
                    let Some(prior) = hop.node_slot.and_then(|s| self.cols[s].as_ref()) else {
                        continue;
                    };

                    if prior[i] != a.nbr {
                        continue;
                    }
                }

                if hop.rejoin_rel {
                    let Some(prior) = hop.rel_slot.and_then(|s| self.cols[s].as_ref()) else {
                        continue;
                    };

                    if prior[i] != a.eidx {
                        continue;
                    }
                }

                if !f.keep(a.eidx, a.nbr) {
                    continue;
                }

                for (s, col) in new_cols.iter_mut().enumerate() {
                    let Some(col) = col else { continue };
                    let v = if hop.rel_slot == Some(s) && !hop.rejoin_rel {
                        a.eidx
                    } else if hop.node_slot == Some(s) && !hop.rejoin_node {
                        a.nbr
                    } else if let Some(prior) = &self.cols[s] {
                        prior[i]
                    } else {
                        // Bound by a LATER hop — no value in this row yet.
                        continue;
                    };

                    col.push(v);
                }

                for (s, col) in new_vals.iter_mut().enumerate() {
                    let (Some(col), Some(prior)) = (col, &self.vals[s]) else {
                        continue;
                    };

                    col.push(prior[i].clone());
                }

                new_endpoint.push(a.nbr);

                if cap.is_some_and(|c| new_endpoint.len() >= c) {
                    break 'rows;
                }

                if new_endpoint.len() as u64 > budget {
                    return Err(TooWide);
                }
            }
        }

        self.cols = new_cols;
        self.vals = new_vals;
        self.endpoint = new_endpoint;
        Ok(())
    }
}
