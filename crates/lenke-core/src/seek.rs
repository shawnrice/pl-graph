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

use crate::graph::{Graph, IdxKey, RangeBound};

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
#[derive(Clone, Debug)]
pub enum Operand {
    Lit(IdxKey),
    Param(usize),
}

impl Operand {
    fn resolve(&self, param: &impl Fn(usize) -> Option<IdxKey>) -> Option<IdxKey> {
        match self {
            Self::Lit(k) => Some(k.clone()),
            Self::Param(slot) => param(*slot),
        }
    }
}

/// One `key OP value` a front end recognized.
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
pub struct ElementSeek {
    /// Which store to seek: edges have their own indexes and key ids.
    pub edge: bool,
    /// Conjunctive — every one must hold, so any single one bounds a candidate
    /// SUPERSET and the caller re-verifies. That is what lets the most selective
    /// one be chosen freely.
    conj: Vec<KeyPredicate>,
    /// Disjunctive — `IN` lists and folded `OR`s of equalities on one key. Every
    /// branch must be seekable or the whole disjunction is not: missing one
    /// branch loses rows, unlike missing a conjunct.
    disj: Vec<(Arc<str>, Vec<Operand>)>,
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

    /// True when nothing seekable was recognized — the caller should scan.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conj.is_empty() && self.disj.is_empty()
    }

    /// A `key OP value` conjunct.
    pub fn push(&mut self, key: Arc<str>, op: SeekOp, operand: Operand) {
        self.conj.push(KeyPredicate { key, op, operand });
    }

    /// A disjunction of equalities on ONE key — an `IN` list, or an `OR` of
    /// equalities the front end folded together.
    ///
    /// A singleton collapses to a conjunct: `u.k IN [$a]` is `u.k = $a`, and
    /// leaving it as a one-branch union would take a different code path for a
    /// predicate that is character-for-character equivalent. That is exactly the
    /// class of divergence this module exists to remove.
    ///
    /// An EMPTY list is not a no-op — `u.k IN []` matches nothing — so it is
    /// kept, and resolves to an empty candidate set rather than a scan.
    pub fn push_any_of(&mut self, key: Arc<str>, values: Vec<Operand>) {
        if let [only] = values.as_slice() {
            self.push(key, SeekOp::Eq, only.clone());
        } else {
            self.disj.push((key, values));
        }
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
    pub fn resolve(
        &self,
        graph: &Graph,
        param: &impl Fn(usize) -> Option<IdxKey>,
    ) -> Option<Vec<u32>> {
        // Take the SMALLEST candidate set among the seekable options rather than
        // the first. A conjunction ANDs necessary conditions, so every candidate
        // set is a valid superset and the smallest is the cheapest correct one.
        // Picking the first is what made `has('country','US').has('ssn',$x)` seed
        // from the wrong side in Gremlin while GQL seeded from the right one.
        //
        // Intersecting them instead was measured to LOSE: building and probing
        // two ~200k halves costs more than the scan it replaces.
        let mut best: Option<Vec<u32>> = None;
        let mut keep = |candidate: Vec<u32>| {
            if best.as_ref().is_none_or(|b| candidate.len() < b.len()) {
                best = Some(candidate);
            }
        };

        for (key, rb) in self.ranges(param) {
            if let Some(ids) = idx_range(graph, &key, &rb, self.edge) {
                keep(ids);
            }
        }

        for (key, values) in &self.disj {
            if let Some(ids) = self.union_of(graph, key, values, param) {
                keep(ids);
            }
        }

        best
    }

    /// Conjuncts folded into one tight [`RangeBound`] per key.
    ///
    /// `x >= 5 AND x <= 9` is one bounded seek, not two; and the caller gets
    /// `5 <= x AND 9 >= x` as the same thing, because the operand order was
    /// normalized away before it ever reached here.
    fn ranges(&self, param: &impl Fn(usize) -> Option<IdxKey>) -> Vec<(Arc<str>, RangeBound)> {
        let mut out: Vec<(Arc<str>, RangeBound)> = Vec::new();

        for p in &self.conj {
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

        out
    }

    /// A disjunction seeds only when EVERY branch does — one unseekable branch
    /// means rows outside the union, so the union is no longer a superset.
    fn union_of(
        &self,
        graph: &Graph,
        key: &str,
        values: &[Operand],
        param: &impl Fn(usize) -> Option<IdxKey>,
    ) -> Option<Vec<u32>> {
        if !idx_indexed(graph, key, self.edge) {
            return None;
        }

        let mut seen: HashSet<u32> = HashSet::new();
        let mut out = Vec::new();

        for v in values {
            // A repeated value must not produce a repeated row: `IN ['a','a']`
            // is `= 'a'`, and the seed is a candidate LIST, not a set.
            for id in idx_eq(graph, key, &v.resolve(param)?, self.edge)? {
                if seen.insert(id) {
                    out.push(id);
                }
            }
        }

        Some(out)
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

pub(crate) fn idx_indexed(graph: &Graph, name: &str, edge: bool) -> bool {
    if edge {
        graph.edge_indexed(name)
    } else {
        graph.vertex_indexed(name)
    }
}

pub(crate) fn idx_eq(graph: &Graph, name: &str, k: &IdxKey, edge: bool) -> Option<Vec<u32>> {
    if edge {
        graph.edges_by_prop(name, k).map(<[u32]>::to_vec)
    } else {
        graph.vertices_by_prop(name, k).map(<[u32]>::to_vec)
    }
}

pub(crate) fn idx_range(
    graph: &Graph,
    name: &str,
    rb: &RangeBound,
    edge: bool,
) -> Option<Vec<u32>> {
    if edge {
        graph.edges_by_prop_range(name, rb)
    } else {
        graph.vertices_by_prop_range(name, rb)
    }
}

#[cfg(test)]
mod tests;
