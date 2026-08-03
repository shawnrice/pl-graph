//! The Gremlin executor: runs a [`Traversal`]'s [`Step`] list over a stream of
//! traversers against the columnar [`Graph`]. Eager (Vec-per-step) — the modest
//! result scale doesn't need lazy iterators, and it keeps step semantics
//! readable. Movement/projection steps extend each traverser's path; filters
//! pass traversers through unchanged. `by()` modulators resolve via [`eval_by`].

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{
    By, Column, Endpoint, GVal, Order, Pop, PropVal, SackOp, Scope, Step, Token, Traversal, P,
};
use crate::graph::{Graph, IdxKey, Value};
use crate::jsonfmt::{push_json_str, push_num};
use crate::seek::{ElementSeek, KeyPredicate, Operand, SeekOp};

/// A hashable projection of a [`GVal`] for O(1) dedup. Mirrors `GVal`'s derived
/// structural equality, with the two `f64` details handled so it matches
/// `PartialEq` exactly: `-0.0`/`+0.0` canonicalize together (they're `==`), and a
/// `NaN` makes the whole key un-hashable — [`dedup_key`] returns `None` and the
/// caller passes the traverser straight through (a `NaN` is never equal to
/// anything, so it can never be a duplicate).
#[derive(Debug, PartialEq, Eq, Hash)]
enum DedupKey {
    Null,
    Bool(bool),
    Num(u64),
    Str(Arc<str>),
    Temporal(crate::temporal::Temporal),
    Vertex(u32),
    Edge(u32),
    List(Vec<Self>),
    Map(Vec<(Self, Self)>),
}

/// Build a hashable dedup key from a `GVal`, or `None` if it contains a `NaN`.
fn dedup_key(v: &GVal) -> Option<DedupKey> {
    // `Record`/`Path` are GQL-only (see `crate::value`); no traversal step can
    // put one in a stream, so there is nothing to key them by.
    Some(match v {
        // GQL-only (see `crate::value`); no traversal step can put one in a
        // stream, so there is nothing to key them by.
        GVal::Record(_) | GVal::Path(_) => return None,
        GVal::Null => DedupKey::Null,
        GVal::Bool(b) => DedupKey::Bool(*b),
        GVal::Num(n) => {
            if n.is_nan() {
                return None;
            }
            // `-0.0 == 0.0`, so collapse both to one bit pattern.
            DedupKey::Num(if *n == 0.0 { 0 } else { n.to_bits() })
        }
        GVal::Str(s) => DedupKey::Str(s.clone()),
        GVal::Temporal(t) => DedupKey::Temporal(*t),
        GVal::Node(id) => DedupKey::Vertex(*id),
        GVal::Edge(id) => DedupKey::Edge(*id),
        GVal::List(xs) => DedupKey::List(xs.iter().map(dedup_key).collect::<Option<_>>()?),
        GVal::Map(kvs) => DedupKey::Map(
            kvs.iter()
                .map(|(k, val)| Some((dedup_key(k)?, dedup_key(val)?)))
                .collect::<Option<_>>()?,
        ),
        // Owner-agnostic (like PartialEq / the TS engine): dedup a property
        // element by its key+value.
        GVal::Property(p) => {
            DedupKey::List(vec![DedupKey::Str(p.key.clone()), dedup_key(&p.value)?])
        }
    })
}

/// Mulberry32 PRNG — a tiny, fast, fully-specified generator. `sample()` uses it
/// with a FIXED seed so the pseudo-random selection is reproducible AND
/// byte-identical with the TS engine (which runs the same algorithm). Same seed +
/// same draw order + same shuffle ⇒ same sample on both engines.
struct Mulberry32 {
    s: u32,
}

impl Mulberry32 {
    fn new(seed: u32) -> Self {
        Self { s: seed }
    }
    /// Next float in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        self.s = self.s.wrapping_add(0x6d2b_79f5);
        let mut t = (self.s ^ (self.s >> 15)).wrapping_mul(1u32 | self.s);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61u32 | t));
        f64::from(t ^ (t >> 14)) / 4_294_967_296.0
    }
}

/// The fixed `sample()` seed, shared with the TS engine for cross-engine parity.
const SAMPLE_SEED: u32 = 0x9e37_79b9;

/// A unit flowing through the pipeline: its value, the path it took, `as(label)`
/// tags (label → accumulated values, for `select` pop), and the repeat loop count.
#[derive(Clone)]
struct Trav {
    val: GVal,
    path: Vec<GVal>,
    /// `as(label)` bindings, label → the values that label took. COPY-ON-WRITE:
    /// every `step`/`with` clones a traverser, and a labelled traversal
    /// (`match()`, `select(Pop.all)`, `repeat(__.as('x')…)`) carries these
    /// through every hop — deep-copying a `String` and a `Vec` per tag per step.
    /// Sharing makes that a refcount bump; the few writers say `make_mut`.
    tags: Arc<Vec<(String, Vec<GVal>)>>,
    loops: usize,
    /// The per-traverser sack (TinkerPop `sack()`). LAZY: `None` until a
    /// `sack(op)` write allocates it; a read before that returns the `withSack`
    /// default (held on `Ctx`) without storing. Boxed so an unused sack costs one
    /// pointer, not an inline `GVal`, and never touches the heap. Split-on-branch
    /// is the plain clone `step`/`with` already do.
    sack: Option<Box<GVal>>,
}

impl Trav {
    fn root(val: GVal) -> Self {
        Self {
            path: vec![val.clone()],
            val,
            tags: Arc::new(Vec::new()),
            loops: 0,
            sack: None,
        }
    }
    /// A successor that moved to `val` (extends path, keeps tags/loops/sack).
    fn step(&self, val: GVal) -> Self {
        // See `TRACK_PATH`: skipped entirely when no step in this run reads it.
        let mut path = if TRACK_PATH.with(Cell::get) {
            self.path.clone()
        } else {
            Vec::new()
        };

        if TRACK_PATH.with(Cell::get) {
            path.push(val.clone());
        }
        Self {
            val,
            path,
            tags: self.tags.clone(),
            loops: self.loops,
            sack: self.sack.clone(),
        }
    }
    /// Same traverser with a replaced value, keeping the existing path tail.
    fn with(&self, val: GVal) -> Self {
        let mut path = self.path.clone();
        path.push(val.clone());
        Self {
            val,
            path,
            tags: self.tags.clone(),
            loops: self.loops,
            sack: self.sack.clone(),
        }
    }
    fn recall(&self, label: &str, pop: Pop) -> Option<GVal> {
        let list = &self.tags.iter().find(|(l, _)| l == label)?.1;
        match pop {
            _ if list.is_empty() => None,
            Pop::First => Some(list[0].clone()),
            Pop::Last => Some(list[list.len() - 1].clone()),
            Pop::All => Some(GVal::list(list.clone())),
        }
    }
}

/// Per-run mutable context: named side-effect bags for `aggregate`/`store`/`cap`,
/// plus `subgraph(key)` accumulators (deduped (vertex ids, edge ids)).
/// Per-`repeat()` cap on the traversers its body produces — a `repeat(both())`
/// with no `until`/`times` on a cyclic/dense graph grows the frontier
/// multiplicatively each level and can exhaust memory. Past this the repeat
/// stops and records `over_budget` so a caller can surface `ResourceExhausted`.
const REPEAT_BUDGET: u64 = 1_000_000;

thread_local! {
    /// Whether this run has to accumulate each traverser's PATH.
    ///
    /// `Trav::step` cloned the whole path vector for every traverser at every
    /// step, so a 3-step traversal over 200k elements paid ~600k allocations to
    /// build a history almost nothing reads. Only five steps read it — `path`,
    /// `simplePath`, `cyclicPath`, `otherV`, `tree` — and skipping the
    /// accumulation when none is present measured 3.6x on `out().values()`
    /// (68.80 -> 19.26 ms) and 1.6x on `out().dedup().count()`.
    ///
    /// The same treatment `sack` already has, for the same reason.
    ///
    /// Set from an ALLOWLIST, so it fails safe: a step must be known both to
    /// leave the path alone AND to nest no sub-traversal before the path can be
    /// dropped. Anything unrecognized — including any step added later — keeps
    /// the accumulation. Getting this backwards would make `path()` silently
    /// return the wrong thing rather than fail.
    static TRACK_PATH: Cell<bool> = const { Cell::new(true) };

    /// Set when evaluation hits a type error — ordering genuinely incomparable
    /// types (a number vs a string, an element vs a scalar) or feeding a
    /// non-number into a numeric aggregate (`sum`/`mean`). TinkerPop raises a
    /// `ClassCastException`; the executor is single-threaded and eager, so a
    /// thread-local flag avoids threading a sink through every leaf helper. It is
    /// reset per `run` and surfaced as `InvalidValue` by `try_run`.
    static TYPE_FAULT: Cell<bool> = const { Cell::new(false) };
}

fn set_type_fault() {
    TYPE_FAULT.with(|f| f.set(true));
}

fn take_type_fault() -> bool {
    TYPE_FAULT.with(|f| f.replace(false))
}

#[derive(Default)]
struct Ctx {
    side: HashMap<String, Vec<GVal>>,
    subgraphs: HashMap<String, (Vec<u32>, Vec<u32>)>,
    /// Set when a `repeat()` hit `REPEAT_BUDGET` and stopped early.
    over_budget: bool,
    /// First mutation/data fault recorded during the run (e.g. `addE()` with an
    /// unresolvable endpoint, or a `fail()` reached). Surfaced by [`try_run`];
    /// ignored by [`run`]. `Cow` so most faults stay `&'static` while `fail()`
    /// can carry its user-supplied message.
    fault: Option<(
        crate::error_codes::ErrorCode,
        std::borrow::Cow<'static, str>,
    )>,
    /// The `withSack(init)` default, set once by the leading `withSack` step.
    /// `None` = no sack configured, so `sack()` faults and NO per-traverser sack
    /// is ever created — the laziness guarantee.
    sack_init: Option<GVal>,
}

/// Run a traversal against `graph`, returning the final traversers' values.
/// Infallible: a runaway `repeat()` stops at the budget and returns its partial
/// frontier. Use [`try_run`] to surface that as a `ResourceExhausted` error.
pub fn run(graph: &mut Graph, t: &Traversal) -> Vec<GVal> {
    let mut ctx = Ctx::default();
    run_collect(graph, &mut ctx, t)
}

/// Like [`run`], but reports a `repeat()` budget overrun as `ResourceExhausted`
/// instead of silently returning a partial result.
pub fn try_run(graph: &mut Graph, t: &Traversal) -> crate::error::CodeResult<Vec<GVal>> {
    let mut ctx = Ctx::default();
    let vals = run_collect(graph, &mut ctx, t);
    if ctx.over_budget {
        return Err(crate::error::CodeError::new(
            crate::error_codes::ErrorCode::ResourceExhausted,
            "repeat() exceeded the traversal budget; add a tighter until()/times()",
        ));
    }
    if take_type_fault() {
        return Err(crate::error::CodeError::new(
            crate::error_codes::ErrorCode::InvalidValue,
            "comparison or numeric aggregation over incomparable/non-numeric values",
        ));
    }
    if let Some((code, msg)) = ctx.fault {
        return Err(crate::error::CodeError::new(code, msg));
    }
    Ok(vals)
}

fn run_collect(graph: &mut Graph, ctx: &mut Ctx, t: &Traversal) -> Vec<GVal> {
    take_type_fault(); // reset any leftover flag from a prior run on this thread
    TRACK_PATH.with(|c| c.set(needs_path(&t.steps)));

    // A seeded plan drops the `V()`/`E()` it started from plus the filters the
    // index answered exactly. Every OTHER filter still runs, including any that
    // preceded them — the seed is only a superset with respect to those.
    if let Some(n) = try_count(graph, &t.steps) {
        return vec![GVal::Num(n)];
    }

    if let Some(vals) = try_values(graph, &t.steps) {
        return vals;
    }

    match index_seed(graph, &t.steps) {
        Some((seed, answered)) if answered.is_empty() => run_steps(graph, ctx, &t.steps[1..], seed),
        Some((seed, answered)) => {
            let rest: Vec<Step> = t.steps[1..]
                .iter()
                .enumerate()
                .filter(|(i, _)| !answered.contains(&(i + 1)))
                .map(|(_, step)| step.clone())
                .collect();

            run_steps(graph, ctx, &rest, seed)
        }
        None => run_steps(graph, ctx, &t.steps, Vec::new()),
    }
    .into_iter()
    .map(|t| t.val)
    .collect()
}

/// A traversal value as an index key — see [`GVal::index_key`], which both
/// engines now share. Gaining the shared version is what gave Gremlin temporal
/// index seeking: this copy had no `Temporal` arm.
fn gval_to_idxkey(v: &GVal) -> Option<IdxKey> {
    v.index_key()
}

/// The smallest string strictly greater than every string with prefix `s`
/// (for `startsWith` → `[s, s⁺)`). `None` ⇒ no upper bound (e.g. empty prefix).
fn prefix_upper(s: &str) -> Option<String> {
    let mut bytes = s.as_bytes().to_vec();
    while let Some(&last) = bytes.last() {
        if last < 0xff {
            *bytes.last_mut().unwrap() = last + 1;
            return String::from_utf8(bytes).ok();
        }
        bytes.pop();
    }
    None
}

/// Candidate elements for a traversal that opens with `V()` / `E()` followed by
/// element filters, via the shared access path in [`crate::seek`].
///
/// Only pure element-filter steps are read. `hasLabel` / `has` / `hasNot` narrow
/// the current element set without changing what a traverser IS, so seeding from
/// any of them and re-running the rest over the seed gives the same answer. A
/// navigating step (`out`, `values`, …) rebinds the traverser, so a `has` after
/// one addresses a different element entirely and must stop the search.
///
/// Returns the seed and how many leading steps the caller may skip: the
/// `V()`/`E()` always, plus any `has` on the key the index answered EXACTLY.
///
/// Re-applying that `has` would be a no-op filter, but not a free one — it is a
/// second pass over the whole seed, which measured a clean 2x on `range` and
/// `between` in `gremlin_index_bench`. Only steps on the exact key are dropped;
/// a `has` on any other key is still doing work, because the seed is only a
/// superset with respect to those.
/// Lower the leading `V()`/`E()` + element-filter run into the shared IR.
///
/// Returns the seek, which steps it captured in full, and how many leading steps
/// it read. Split out so a TERMINAL (`count()`) can reuse the same lowering
/// rather than re-deriving it.
fn lower_prefix(graph: &Graph, steps: &[Step]) -> Option<(ElementSeek, Vec<usize>, usize, bool)> {
    let is_edge = match steps.first()? {
        Step::V(ids) if ids.is_empty() => false,
        Step::E(ids) if ids.is_empty() => true,
        _ => return None,
    };
    let mut seek = ElementSeek::same_kind(is_edge);
    let mut captured: Vec<usize> = Vec::new();
    let mut read = 1;

    for (i, step) in steps.iter().enumerate().skip(1) {
        match step {
            Step::Has(key, pred) => {
                if lower_predicate(key, pred, &mut seek) {
                    captured.push(i);
                }
            }
            Step::HasLabel(labels) if seek.labels().is_none() => {
                let ids: Vec<u32> = if is_edge {
                    labels.iter().filter_map(|l| graph.etype.get(l)).collect()
                } else {
                    labels.iter().filter_map(|l| graph.labels.get(l)).collect()
                };

                if ids.len() == labels.len() {
                    seek.set_labels(crate::seek::LabelRule::First, ids);
                    captured.push(i);
                } else if !labels.is_empty() {
                    // A name that resolved to nothing matches nothing; say so in
                    // the IR rather than letting the step scan for it.
                    seek.set_labels(crate::seek::LabelRule::First, Vec::new());
                    captured.push(i);
                }
            }
            Step::HasLabel(..) | Step::HasNot(..) => {}
            _ => break,
        }

        read = i + 1;
    }

    Some((seek, captured, read, is_edge))
}

/// Steps that neither READ the traverser path nor nest a sub-traversal.
///
/// An allowlist on purpose — see `TRACK_PATH`. A step missing from here only
/// costs the path accumulation it would have had anyway; a step wrongly added
/// would corrupt `path()`.
///
/// `Dedupe` qualifies only with an EMPTY `bys`: a `by()` modulator can hold a
/// whole sub-traversal, and that sub-traversal may read the path. Its `labels`
/// are fine either way — they read the tag map, not the path.
fn path_free(step: &Step) -> bool {
    // Steps that carry sub-traversals are path-free exactly when their bodies
    // are: `repeat(out())` needs no history, `repeat(simplePath())` does.
    // Recursing here is what lets the common looping shapes off the path tax at
    // all — as a flat list they could never be listed, so every one of them paid.
    let subs: &[&Traversal] = match step {
        Step::Where(t)
        | Step::Not(t)
        | Step::Optional(t)
        | Step::Local(t)
        | Step::Map(t)
        | Step::FlatMap(t)
        | Step::SideEffect(t) => &[t],
        Step::And(ts) | Step::Or(ts) | Step::Union(ts) | Step::Coalesce(ts) => {
            return ts.iter().all(|t| t.steps.iter().all(path_free));
        }
        Step::Choose { test, then_, else_ } => {
            return [Some(test), Some(then_), else_.as_ref()]
                .into_iter()
                .flatten()
                .all(|t| t.steps.iter().all(path_free));
        }
        Step::Repeat { body, until, .. } => {
            return [Some(body), until.as_ref()]
                .into_iter()
                .flatten()
                .all(|t| t.steps.iter().all(path_free));
        }
        _ => &[],
    };

    if let [t] = subs {
        return t.steps.iter().all(path_free);
    }

    matches!(
        step,
        Step::V(_)
            | Step::E(_)
            | Step::Has(..)
            | Step::HasLabel(..)
            | Step::HasNot(..)
            | Step::HasKey(..)
            | Step::HasId(..)
            | Step::HasValue(..)
            | Step::Out(..)
            | Step::In(..)
            | Step::Both(..)
            | Step::OutE(..)
            | Step::InE(..)
            | Step::BothE(..)
            | Step::Values(..)
            | Step::Count(_)
            | Step::Limit(..)
            | Step::Skip(..)
            | Step::Id
            | Step::Label
            | Step::Sum(_)
            | Step::Mean(_)
            | Step::Min(_)
            | Step::Max(_)
            // Only FIVE steps actually read `Trav::path`: OtherV (which vertex did
            // we come from), SimplePath / CyclicPath (repeat check), Path (emits
            // it) and Sack. Everything below was paying a per-traverser path
            // clone for history nothing was going to read — 63% of the suite's
            // traversals tracked paths, and this is why.
            | Step::Order(..)
            | Step::Fold
            | Step::Unfold
            | Step::Range(..)
            | Step::Tail(..)
            | Step::Is(_)
            | Step::WherePred(_)
            | Step::Group(..)
            | Step::GroupCount(..)
            | Step::Project(..)
            | Step::ValueMap(..)
            | Step::PropertyMap(..)
            | Step::Properties(..)
            | Step::ElementMap(..)
            | Step::Value
            | Step::Constant(_)
            | Step::Barrier
            | Step::Sample(_)
            | Step::Aggregate(_)
            | Step::Store(_)
            | Step::Cap(_)
    ) || matches!(step, Step::Dedupe { bys, .. } if bys.is_empty())
}

/// Whether this traversal needs per-traverser path history at all.
fn needs_path(steps: &[Step]) -> bool {
    !steps.iter().all(path_free)
}

/// The whole live element set, when nothing narrows it.
fn universe(graph: &Graph, is_edge: bool) -> Vec<u32> {
    if is_edge {
        (0..graph.e_src.len() as u32)
            .filter(|&e| graph.is_edge_live(e))
            .collect()
    } else {
        graph.vertex_indices().collect()
    }
}

/// One lowered navigation hop: a direction and the edge types to follow.
///
/// `None` for the types means a name that resolved to nothing — which matches
/// nothing, and must not be confused with an EMPTY list, which means any type.
type Hop = (crate::seek::Dir, Option<Vec<u32>>);

/// Parse a chain of navigation steps into IR hops, returning what follows.
///
/// `outE(L).inV()` IS `out(L)` — the edge step selects out-edges and the vertex
/// step takes their far end — so both spellings lower to the SAME hop. That is
/// the point of having one IR: the optimization is not written twice, and the
/// choice between the spellings stops mattering.
///
/// `bothE(L).otherV()` is deliberately NOT folded: `otherV` reads the traverser
/// PATH to know which end it arrived from, so it is not a pure function of the
/// edge.
fn lower_hops<'a>(graph: &Graph, mut rest: &'a [Step]) -> (Vec<Hop>, &'a [Step]) {
    let mut hops = Vec::new();

    loop {
        // `repeat(<hops>).times(n)` is those hops n times over. Only the plain
        // counted form: `until` and `emit` decide PER TRAVERSER whether to stop
        // or yield, which is a predicate over the stream rather than a fixed
        // shape, and an unbounded `repeat` has no length to unroll.
        if let [Step::Repeat {
            body,
            times: Some(n),
            until: None,
            emit: None,
            ..
        }, tail @ ..] = rest
        {
            let (inner, leftover) = lower_hops(graph, &body.steps);

            // The body must be hops and NOTHING else — a filter or a projection
            // inside it changes what each repetition yields.
            if leftover.is_empty() && !inner.is_empty() {
                for _ in 0..*n {
                    hops.extend(inner.iter().cloned());
                }

                rest = tail;
                continue;
            }

            break;
        }

        let (dir, labels, tail) = match rest {
            [Step::Out(l), t @ ..] => (crate::seek::Dir::Out, l, t),
            [Step::In(l), t @ ..] => (crate::seek::Dir::In, l, t),
            [Step::Both(l), t @ ..] => (crate::seek::Dir::Both, l, t),
            [Step::OutE(l), Step::InV, t @ ..] => (crate::seek::Dir::Out, l, t),
            [Step::InE(l), Step::OutV, t @ ..] => (crate::seek::Dir::In, l, t),
            _ => break,
        };
        let etypes: Vec<u32> = labels.iter().filter_map(|l| graph.etype.get(l)).collect();

        // A type name that resolved to nothing matches nothing; an EMPTY list
        // means "any type", so the two must not be confused. `None` carries the
        // former so the caller can stop rather than expand everything.
        hops.push((dir, (etypes.len() == labels.len()).then_some(etypes)));
        rest = tail;
    }

    (hops, rest)
}

/// The element ids a lowered prefix plus expansion chain produces, and whatever
/// steps are left over.
///
/// Shared by the terminals so each does not re-derive the walk.
fn lowered_ids<'a>(graph: &Graph, steps: &'a [Step]) -> Option<(Vec<u32>, &'a [Step], bool)> {
    let (seek, captured, read, is_edge) = lower_prefix(graph, steps)?;

    // Every filter in the run must have been captured — an uncaptured one can
    // only run over a stream.
    if captured.len() != read - 1 {
        return None;
    }

    let no_params = |_: usize| None;

    if !seek.types_agree(graph, &no_params) {
        return None;
    }
    if !(seek.columnar(graph, &no_params)
        || (seek.conj_is_empty() && (seek.labels().is_some() || read == 1)))
    {
        return None;
    }

    // From an EDGE frontier only a terminal may follow. `E().outE(R).inV()` would
    // otherwise be read as a hop off the edge ids, which are vertex ids to
    // `lower_hops` and to nothing else. An allowlist rather than a denylist: a
    // step missing from here only declines to lower, a navigating step wrongly
    // added walks the wrong adjacency and answers.
    if is_edge
        && !matches!(
            steps.get(read),
            None | Some(
                Step::Values(_)
                    | Step::Count(_)
                    | Step::Id
                    | Step::Label
                    | Step::Fold
                    | Step::Sum(_)
                    | Step::Min(_)
                    | Step::Max(_)
                    | Step::Mean(_)
            )
        )
    {
        return None; // edge-to-edge navigation is not this shape
    }

    let mut ids = seek.scan(graph, &no_params, || universe(graph, is_edge));
    let (hops, rest) = lower_hops(graph, &steps[read..]);
    let edges = is_edge && hops.is_empty();

    for (dir, etypes) in &hops {
        let Some(etypes) = etypes else {
            return Some((Vec::new(), rest, edges));
        };

        ids = crate::seek::expand(graph, &ids, *dir, etypes, crate::seek::SelfLoops::Twice);
    }

    // A trailing edge step lands on the EDGE rather than its far end, turning the
    // frontier into edge ids — which is what lets `outE(R).count()` and
    // `outE(R).values(w)` answer from the IR too.
    if !edges {
        let (dir, labels, tail) = match rest {
            [Step::OutE(l), t @ ..] => (crate::seek::Dir::Out, l, t),
            [Step::InE(l), t @ ..] => (crate::seek::Dir::In, l, t),
            [Step::BothE(l), t @ ..] => (crate::seek::Dir::Both, l, t),
            _ => return Some((ids, rest, edges)),
        };
        let etypes: Vec<u32> = labels.iter().filter_map(|l| graph.etype.get(l)).collect();

        if etypes.len() != labels.len() {
            return Some((Vec::new(), tail, true));
        }

        return Some((
            crate::seek::expand_edges(graph, &ids, dir, &etypes, crate::seek::SelfLoops::Twice),
            tail,
            true,
        ));
    }

    Some((ids, rest, edges))
}

/// The value a lowered frontier id IS, so a terminal can reuse the very
/// projections the streaming path applies (`elem_id`, `elem_label`) instead of
/// re-deriving them and drifting.
fn frontier_val(id: u32, is_edge: bool) -> GVal {
    if is_edge {
        GVal::Edge(id)
    } else {
        GVal::Node(id)
    }
}

/// `V()/E() … <terminal>` answered from the IR, with no traverser built for any
/// of the 200k rows these used to allocate one for.
///
/// Every arm here must agree with running the same steps over a stream — the
/// tests in `index_seed_tests` assert exactly that, terminal by terminal.
///
/// Over 20k vertices / 60k edges (`min` of 9, `tmp_terminal_bench`):
///
/// ```text
///                                    stream    lowered
///   V().hasLabel(P).id()             1012 us     110 us
///   V().hasLabel(P).label()          1001         105
///   V().id()                         1877         178
///   V().out().id()                  10857         651
///   E().id()                         9332        2590
///   V().values(n).sum()               953          39
///   V().values(n).min()               956          72
///   E().values(w).sum()              2816         113
///   V().out().values(n).sum()        3218         213
///   V().values(n).is(gt).count()     1870          55
///   V().hasLabel(P).fold()            330          53
///   V().values(n).fold()             1877          84
///   V().hasLabel(P).count(local)     1147          60
/// ```
///
/// `E().id()` is the one row that stays expensive, and the lowering is not why:
/// an edge id is materialized as an owned `String` and then re-allocated into an
/// `Arc<str>` PER EDGE, which is 60k allocation pairs that the frontier walk
/// cannot remove. Interning edge ids the way vertex ids already are is the
/// change that would move it, and it is a storage change, not a planning one.
fn try_values(graph: &Graph, steps: &[Step]) -> Option<Vec<GVal>> {
    let (ids, rest, is_edge) = lowered_ids(graph, steps)?;

    match rest {
        // `id()` / `label()` are pure per-element projections of the frontier —
        // the same shape as `values(k)`, reading the id/label dictionaries
        // instead of a property column.
        [Step::Id] => Some(
            ids.iter()
                .map(|&id| elem_id(graph, &frontier_val(id, is_edge)))
                .collect(),
        ),
        [Step::Label] => Some(
            ids.iter()
                .map(|&id| elem_label(graph, &frontier_val(id, is_edge)))
                .collect(),
        ),
        [Step::Fold] => Some(vec![GVal::List(
            ids.into_iter()
                .map(|id| frontier_val(id, is_edge))
                .collect(),
        )]),
        // `count(local)` counts a value's own elements, and a graph element is
        // not iterable — `local_elems` wraps it in a singleton. So it is 1 per
        // row, whatever the row is.
        [Step::Count(Scope::Local)] => Some(vec![GVal::Num(1.0); ids.len()]),
        [Step::Values(keys), tail @ ..] => column_terminal(graph, &ids, is_edge, keys, tail),
        _ => None,
    }
}

/// `values(k)` and whatever follows it, answered from the typed column.
fn column_terminal(
    graph: &Graph,
    ids: &[u32],
    is_edge: bool,
    keys: &[String],
    tail: &[Step],
) -> Option<Vec<GVal>> {
    // One key only: `values()` needs a per-element key list, and a multi-key call
    // interleaves columns per element rather than reading one.
    let [key] = keys else {
        return None;
    };
    let store = if is_edge {
        &graph.edge_props
    } else {
        &graph.props
    };
    // Split an optional `is(P)` off the front. `is` filters the CURRENT value,
    // which after `values(k)` is a column value — so there, and only there, it is
    // a column predicate rather than a test on a graph element.
    let (filter, tail) = match tail {
        [Step::Is(p), t @ ..] => (Some(p), t),
        t => (None, t),
    };
    let Some(kid) = store.keys.get(key) else {
        // No element ever carried this key, so `values(k)` emits nothing and
        // every terminal folds the empty stream. `is` cannot change that.
        return empty_column_terminal(tail);
    };

    // A homogeneous NUMBER column answers the filter and every numeric aggregate
    // straight from `data`. Any other column type is left to the stream: a
    // `Str`/`Bool`/`Temporal`/`Mixed` value under `sum()`/`mean()` is a type
    // FAULT rather than a skipped row, and answering it here would make the
    // lowering observable.
    //
    // NOT lowered: `min()`/`max()` over a `Str` column. It is well defined
    // (`values('k').min()` is the lexicographic minimum) but a `Str` column holds
    // INTERNED IDS, whose numeric order is insertion order and has nothing to do
    // with the text — so every comparison needs a dictionary lookup, which is the
    // per-value work the stream already does. There is no column read to win.
    if let Some(crate::graph::Column::Num { data, present }) = store.cols.get(kid as usize) {
        let mut nums: Vec<f64> = Vec::with_capacity(ids.len());

        for &id in ids {
            let i = id as usize;

            if present.get(i) {
                nums.push(data[i]);
            }
        }

        if let Some(out) = num_column_terminal(&nums, filter, tail) {
            return Some(out);
        }

        // Fall through: a shape the numeric path declined (a non-numeric `is`, a
        // terminal it does not cover) may still be a plain projection below.
    }

    if filter.is_some() {
        return None; // a filter this layer could not express has to run per value
    }

    let mut out = Vec::with_capacity(ids.len());

    for &id in ids {
        let i = id as usize;

        // Gate on PRESENCE, not value != Null: a present null rides through.
        if store.is_present_id(i, kid) {
            out.push(value_to_gval(store.value_id(i, kid, &graph.strs)));
        }
    }

    match tail {
        [] => Some(out),
        [Step::Fold] => Some(vec![GVal::list(out)]),
        #[allow(clippy::cast_precision_loss)]
        [Step::Count(Scope::Global)] => Some(vec![GVal::Num(out.len() as f64)]),
        _ => None,
    }
}

/// The terminals over a `values(k)` whose key no element carries — an EMPTY
/// stream, which each aggregate folds its own way. `sum`/`mean`/`min`/`max` over
/// nothing is `null`, not `0`.
fn empty_column_terminal(tail: &[Step]) -> Option<Vec<GVal>> {
    match tail {
        [] | [Step::Count(Scope::Local)] => Some(Vec::new()),
        [Step::Fold] => Some(vec![GVal::list(Vec::new())]),
        [Step::Count(Scope::Global)] => Some(vec![GVal::Num(0.0)]),
        [Step::Sum(Scope::Global) | Step::Mean(Scope::Global)] => Some(vec![GVal::Null]),
        [Step::Min(Scope::Global) | Step::Max(Scope::Global)] => Some(vec![GVal::Null]),
        _ => None,
    }
}

/// A lowered `is(P)` over a number column: the test, plus whether a `NaN` would
/// make the streaming path FAULT rather than answer.
struct NumTest<'a> {
    test: Box<dyn Fn(f64) -> bool + 'a>,
    /// `cmp_or_fault` flags any comparison that returns no ordering, and
    /// `partial_cmp` returns none for `NaN`. So an ORDERING predicate over a
    /// column holding a `NaN` throws through the stream, and answering it from
    /// the column would swallow that. Equality never orders, so it never faults.
    faults_on_nan: bool,
}

/// `is(P)` after `values(k)` as a predicate over `f64` — `None` when it is not
/// one, which is most of them.
///
/// Only where the answer is unambiguous. An ordering against a NON-number is a
/// type fault in `cmp_or_fault`, so those decline outright rather than being
/// answered as "no rows".
fn num_test(p: &P) -> Option<NumTest<'_>> {
    let n = |v: &GVal| match v {
        GVal::Num(x) => Some(*x),
        _ => None,
    };
    let eq = |test: Box<dyn Fn(f64) -> bool>| {
        Some(NumTest {
            test,
            faults_on_nan: false,
        })
    };
    let ord = |test: Box<dyn Fn(f64) -> bool>| {
        Some(NumTest {
            test,
            faults_on_nan: true,
        })
    };

    match p {
        P::Eq(t) => {
            let t = n(t)?;

            eq(Box::new(move |x| x == t))
        }
        P::Neq(t) => {
            let t = n(t)?;

            eq(Box::new(move |x| x != t))
        }
        P::Gt(t) => {
            let t = n(t)?;

            ord(Box::new(move |x| x > t))
        }
        P::Gte(t) => {
            let t = n(t)?;

            ord(Box::new(move |x| x >= t))
        }
        P::Lt(t) => {
            let t = n(t)?;

            ord(Box::new(move |x| x < t))
        }
        P::Lte(t) => {
            let t = n(t)?;

            ord(Box::new(move |x| x <= t))
        }
        // Half-open, like `p_matches`: `>= lo` and `< hi`.
        P::Between(lo, hi) => {
            let (lo, hi) = (n(lo)?, n(hi)?);

            ord(Box::new(move |x| x >= lo && x < hi))
        }
        P::Inside(lo, hi) => {
            let (lo, hi) = (n(lo)?, n(hi)?);

            ord(Box::new(move |x| x > lo && x < hi))
        }
        P::Outside(lo, hi) => {
            let (lo, hi) = (n(lo)?, n(hi)?);

            ord(Box::new(move |x| x < lo || x > hi))
        }
        // `within`/`without` test EQUALITY against each member, so a non-numeric
        // member simply never matches a number — but requiring all-numeric keeps
        // the reasoning local, and a mixed list is not a shape worth chasing.
        P::Within(vs) => {
            let vs: Vec<f64> = vs.iter().map(n).collect::<Option<_>>()?;

            eq(Box::new(move |x| vs.contains(&x)))
        }
        P::Without(vs) => {
            let vs: Vec<f64> = vs.iter().map(n).collect::<Option<_>>()?;

            eq(Box::new(move |x| !vs.contains(&x)))
        }
        // `containing`, `regex`, `not(…)`, … are not a numeric range.
        _ => None,
    }
}

/// The numeric terminals over the present values of a number column.
///
/// `None` declines — a shape this path does not cover, or one where lowering it
/// would change whether the query throws.
#[allow(clippy::cast_precision_loss)]
fn num_column_terminal(nums: &[f64], filter: Option<&P>, tail: &[Step]) -> Option<Vec<GVal>> {
    let owned: Vec<f64>;
    let nums: &[f64] = match filter {
        None => nums,
        Some(p) => {
            let t = num_test(p)?;

            if t.faults_on_nan && nums.iter().any(|x| x.is_nan()) {
                return None; // the stream faults here; do not answer instead
            }

            owned = nums.iter().copied().filter(|&x| (t.test)(x)).collect();
            &owned
        }
    };
    // `fold_extreme` walks the stream keeping the first value that compares the
    // wanted way, so a tie keeps the EARLIER one — which for `-0.0` vs `0.0` is
    // observable. Reproduce the fold rather than calling `f64::min`.
    let extreme = |want: Ordering| {
        // A NaN makes `cmp_or_fault` flag a type fault mid-fold, so decline.
        if nums.iter().any(|x| x.is_nan()) {
            return None;
        }

        Some(nums.iter().fold(GVal::Null, |best, &x| match best {
            GVal::Num(b) if x.partial_cmp(&b) != Some(want) => GVal::Num(b),
            _ => GVal::Num(x),
        }))
    };
    // An empty stream folds to a single `null` for every numeric aggregate — not
    // to `0`, and not to no rows.
    let fold = |f: &dyn Fn(&[f64]) -> f64| {
        if nums.is_empty() {
            GVal::Null
        } else {
            GVal::Num(f(nums))
        }
    };

    match tail {
        [] => Some(nums.iter().copied().map(GVal::Num).collect()),
        [Step::Fold] => Some(vec![GVal::List(
            nums.iter().copied().map(GVal::Num).collect(),
        )]),
        [Step::Count(Scope::Global)] => Some(vec![GVal::Num(nums.len() as f64)]),
        [Step::Count(Scope::Local)] => Some(vec![GVal::Num(1.0); nums.len()]),
        // Summed in FRONTIER order, which is the order the stream would have
        // visited them in: floating-point addition is not associative, so a
        // different order is a different answer and the TS engine's must match.
        [Step::Sum(Scope::Global)] => Some(vec![fold(&|ns| ns.iter().sum())]),
        [Step::Mean(Scope::Global)] => {
            Some(vec![fold(&|ns| ns.iter().sum::<f64>() / ns.len() as f64)])
        }
        [Step::Min(Scope::Global)] => Some(vec![extreme(Ordering::Less)?]),
        [Step::Max(Scope::Global)] => Some(vec![extreme(Ordering::Greater)?]),
        _ => None,
    }
}

/// `V()/E() … count()` answered from the IR, with no traverser built at all.
///
/// The count was costing one `Trav` allocation per element purely to take a
/// length afterwards. Every filter must be captured — an uncaptured one still
/// has to run, and it can only run over a stream.
fn try_count(graph: &Graph, steps: &[Step]) -> Option<f64> {
    let (seek, captured, read, is_edge) = lower_prefix(graph, steps)?;

    // Every filter in the run must have been captured — an uncaptured one can
    // only run over a stream.
    if captured.len() != read - 1 {
        return None;
    }

    let no_params = |_: usize| None;

    if !seek.types_agree(graph, &no_params) {
        return None;
    }
    if !(seek.columnar(graph, &no_params)
        || (seek.conj_is_empty() && (seek.labels().is_some() || read == 1)))
    {
        return None;
    }

    // A CHAIN of expansions, then the count. Each hop keeps duplicates, because
    // each is its own traverser; only the last hop can skip materializing.
    let (hops, mut rest) = lower_hops(graph, &steps[read..]);

    if is_edge && !hops.is_empty() {
        return None;
    }

    // An unresolvable type name mid-chain matches nothing.
    let mut resolved = Vec::with_capacity(hops.len());

    for (dir, etypes) in hops {
        match etypes {
            Some(e) => resolved.push((dir, e)),
            None => return Some(0.0),
        }
    }

    let hops = resolved;

    // An optional plain `dedup()` before the count. Only the argument-free form:
    // `dedup('a')` keys on tags and `dedup().by(..)` on a sub-traversal, neither
    // of which is an element identity.
    let mut distinct = false;

    if let [Step::Dedupe { labels, bys }, tail @ ..] = rest {
        if labels.is_empty() && bys.is_empty() {
            distinct = true;
            rest = tail;
        }
    }

    // A trailing edge step lands on the EDGE rather than its far end, so
    // `outE(R).count()` counts edges. It has to come before the count check
    // because it consumes a step.
    let mut edge_tail: Option<(crate::seek::Dir, Vec<u32>)> = None;

    if !is_edge {
        if let [nav @ (Step::OutE(l) | Step::InE(l) | Step::BothE(l)), tail @ ..] = rest {
            let dir = match nav {
                Step::OutE(_) => crate::seek::Dir::Out,
                Step::InE(_) => crate::seek::Dir::In,
                _ => crate::seek::Dir::Both,
            };
            let etypes: Vec<u32> = l.iter().filter_map(|n| graph.etype.get(n)).collect();

            if etypes.len() != l.len() {
                return Some(0.0);
            }

            edge_tail = Some((dir, etypes));
            rest = tail;
        }
    }

    if !matches!(rest, [Step::Count(Scope::Global)]) {
        return None;
    }

    let mut ids = seek.scan(graph, &no_params, || universe(graph, is_edge));

    if let Some((dir, etypes)) = edge_tail {
        for (d, e) in &hops {
            ids = crate::seek::expand(graph, &ids, *d, e, crate::seek::SelfLoops::Twice);
        }

        let edges =
            crate::seek::expand_edges(graph, &ids, dir, &etypes, crate::seek::SelfLoops::Twice);

        #[allow(clippy::cast_precision_loss)]
        return Some(if distinct {
            let mut seen: HashSet<u32> = HashSet::with_capacity(edges.len());
            let mut e = edges;

            e.retain(|&id| seen.insert(id));
            e.len() as f64
        } else {
            edges.len() as f64
        });
    }

    // With a dedup the last hop has to be materialized to be deduplicated; without
    // one it can be counted in place.
    #[allow(clippy::cast_precision_loss)]
    if distinct {
        for (d, e) in &hops {
            ids = crate::seek::expand(graph, &ids, *d, e, crate::seek::SelfLoops::Twice);
        }

        // Element identity is the dense index, so distinctness is a set of u32 —
        // no key projection, no per-traverser allocation.
        let mut seen: HashSet<u32> = HashSet::with_capacity(ids.len());

        ids.retain(|&id| seen.insert(id));

        return Some(ids.len() as f64);
    }

    #[allow(clippy::cast_precision_loss)]
    match hops.split_last() {
        // No expansion: the filtered set itself.
        None => Some(ids.len() as f64),
        // Walk every hop but the last, then COUNT the last without building it.
        Some(((dir, etypes), init)) => {
            for (d, e) in init {
                ids = crate::seek::expand(graph, &ids, *d, e, crate::seek::SelfLoops::Twice);
            }

            Some(
                crate::seek::expand_count(graph, &ids, *dir, etypes, crate::seek::SelfLoops::Twice)
                    as f64,
            )
        }
    }
}

fn index_seed(graph: &Graph, steps: &[Step]) -> Option<(Vec<Trav>, Vec<usize>)> {
    let is_edge = match steps.first()? {
        Step::V(ids) if ids.is_empty() => false,
        Step::E(ids) if ids.is_empty() => true,
        _ => return None,
    };
    let mut seek = ElementSeek::same_kind(is_edge);
    // Only steps whose predicate was captured IN FULL may be dropped later.
    let mut captured: Vec<usize> = Vec::new();

    for (i, step) in steps.iter().enumerate().skip(1) {
        match step {
            Step::Has(key, pred) => {
                if lower_predicate(key, pred, &mut seek) {
                    captured.push(i);
                }
            }
            // `hasLabel` lowers into the IR too, under the FIRST-label rule —
            // TinkerPop reads `vertex_labels(i).first()`, unlike GQL's any-label
            // `(n:Person)`. Carrying the rule as data is what lets the bucket
            // seeding below serve both. Only the first `hasLabel` is taken: a
            // second one intersects, which the flat id list cannot express, so it
            // stays a step.
            Step::HasLabel(labels) if seek.labels().is_none() => {
                let ids: Vec<u32> = if is_edge {
                    labels.iter().filter_map(|l| graph.etype.get(l)).collect()
                } else {
                    labels.iter().filter_map(|l| graph.labels.get(l)).collect()
                };

                // A name that resolved to nothing matches nothing, which the
                // empty list already says — but only if EVERY name was unknown.
                if ids.len() == labels.len() {
                    seek.set_labels(crate::seek::LabelRule::First, ids);
                    captured.push(i);
                }
            }
            Step::HasLabel(..) | Step::HasNot(..) => {}
            _ => break,
        }
    }

    // Gremlin values are already values — there are no parameter slots to bind.
    let no_params = |_: usize| None;

    // A type-mismatched comparison must reach the per-step path, which raises the
    // fault. Seeding would answer the same ROWS and swallow the error, so an
    // index would change whether the query throws.
    if !seek.types_agree(graph, &no_params) {
        return None;
    }

    // Every leading `has` answerable straight from a typed column: take the whole
    // prefix through the SHARED columnar filter and drop those steps, instead of
    // walking one traverser at a time. `V().has('age', gt(50)).count()` was 15x
    // GQL's equivalent, and the difference was entirely this.
    //
    // `columnar` declines whenever a comparison could be cross-type, because that
    // is a type FAULT here and three-valued UNKNOWN in GQL — neither survives a
    // yes/no filter, so those keep the per-step path that gets them right.
    // A label-only seek is always answerable — no column types involved.
    if seek.columnar(graph, &no_params) || (seek.conj_is_empty() && seek.labels().is_some()) {
        let ids = seek.scan(graph, &no_params, || {
            if is_edge {
                (0..graph.e_src.len() as u32)
                    .filter(|&e| graph.is_edge_live(e))
                    .collect()
            } else {
                graph.vertex_indices().collect()
            }
        });

        return Some((
            ids.into_iter()
                .map(|id| {
                    Trav::root(if is_edge {
                        GVal::Edge(id)
                    } else {
                        GVal::Node(id)
                    })
                })
                .collect(),
            captured,
        ));
    }

    let seeded = seek.resolve_seeded(graph, &no_params)?;
    let answered: Vec<usize> = seeded.exact_key.map_or_else(Vec::new, |key| {
        steps
            .iter()
            .enumerate()
            .skip(1)
            .take_while(|(_, s)| matches!(s, Step::Has(..) | Step::HasLabel(..) | Step::HasNot(..)))
            .filter(|(_, s)| matches!(s, Step::Has(k, _) if **k == *key))
            .map(|(i, _)| i)
            .collect()
    });

    Some((
        seeded
            .ids
            .into_iter()
            .map(|id| {
                Trav::root(if is_edge {
                    GVal::Edge(id)
                } else {
                    GVal::Node(id)
                })
            })
            .collect(),
        answered,
    ))
}

/// One `has(key, P)` as constraints on the shared access path.
///
/// This is where Gremlin's predicate vocabulary meets a plain index: `between`
/// is two bounds on one key, `outside` is a UNION of two half-open ranges (a
/// disjunction, not a range — as a conjunction it would be the empty
/// intersection, the opposite of what it means), and `startsWith` is a prefix
/// range, `>= prefix AND < prefix_upper`. That last one is the form GQL's
/// `starts_with` still lacks and now stands to inherit for free.
///
/// Returns whether the predicate was captured IN FULL. A `false` means the seek
/// only approximates it (or ignores it), so the step must still run — dropping
/// it would silently lose the filter. `neq` and the text predicates land here.
fn lower_predicate(key: &str, pred: &P, seek: &mut ElementSeek) -> bool {
    let key: Arc<str> = Arc::from(key);
    let lit = |v: &GVal| gval_to_idxkey(v).map(Operand::Lit);
    let mut one = |op: SeekOp, v: &GVal| {
        if let Some(o) = lit(v) {
            seek.push(key.clone(), op, o);
        }
    };

    match pred {
        P::Eq(v) => one(SeekOp::Eq, v),
        P::Gt(v) => one(SeekOp::Gt, v),
        P::Gte(v) => one(SeekOp::Ge, v),
        P::Lt(v) => one(SeekOp::Lt, v),
        P::Lte(v) => one(SeekOp::Le, v),
        P::Between(lo, hi) => {
            one(SeekOp::Ge, lo);
            one(SeekOp::Lt, hi);
        }
        P::Inside(lo, hi) => {
            one(SeekOp::Gt, lo);
            one(SeekOp::Lt, hi);
        }
        P::Outside(lo, hi) => {
            let (Some(lo), Some(hi)) = (lit(lo), lit(hi)) else {
                return false;
            };

            seek.push_branches(vec![
                vec![KeyPredicate {
                    key: key.clone(),
                    op: SeekOp::Lt,
                    operand: lo,
                }],
                vec![KeyPredicate {
                    key,
                    op: SeekOp::Gt,
                    operand: hi,
                }],
            ]);
        }
        P::Within(vs) => {
            let Some(values) = vs.iter().map(lit).collect::<Option<Vec<_>>>() else {
                return false;
            };

            // Deduped by the shared layer. The hand-rolled version was not, so
            // `within('a', 'a')` returned the element twice — a duplicate
            // candidate becomes a duplicate ROW, which is a wrong answer.
            seek.push_any_of(key, values);
        }
        P::StartsWith(prefix) => {
            seek.push(
                key.clone(),
                SeekOp::Ge,
                Operand::Lit(IdxKey::Str(prefix.as_str().into())),
            );

            // No upper bound when the prefix is all-maximal: everything at or
            // after it starts with it.
            if let Some(upper) = prefix_upper(prefix) {
                seek.push(
                    key,
                    SeekOp::Lt,
                    Operand::Lit(IdxKey::Str(upper.as_str().into())),
                );
            }
        }
        // `neq`, `without`, `containing`, … enumerate a complement, which no
        // point or range seek covers.
        _ => return false,
    }

    true
}

/// Serialize traversal results to a JSON array string — the FFI carrier. Graph
/// elements become `{"id":…,"label":…}`; lists → arrays; maps → objects.
/// Hand-rolled (no `serde_json`) via the shared [`crate::jsonfmt`] primitives.
pub fn results_to_json(graph: &Graph, vals: &[GVal]) -> String {
    let mut out = String::new();
    out.push('[');
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_gval(&mut out, graph, v);
    }
    out.push(']');
    out
}

/// Serialize a query-result `Value` — which, unlike a stored property, may be a
/// `Map` (a graph element `{id, labels, properties}`) — to JSON. Scalars, lists,
/// and temporals go through the shared `codec::push_value` (so numbers, strings,
/// and tagged temporals match GQL exactly); a `Map` becomes a JSON object.
fn push_result_value(out: &mut String, v: &Value) {
    match v {
        Value::Map(pairs) => {
            out.push('{');
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_json_str(out, k);
                out.push(':');
                push_result_value(out, val);
            }
            out.push('}');
        }
        other => crate::codec::push_value(out, other),
    }
}

fn write_gval(out: &mut String, graph: &Graph, v: &GVal) {
    match v {
        GVal::Null => out.push_str("null"),
        GVal::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        GVal::Num(n) => push_num(out, *n),
        GVal::Str(s) => push_json_str(out, s),
        // A temporal renders as the tagged `{"@date":…}` form (`values()`,
        // `valueMap()`, `project()`, …) — byte-identical to GQL and TS, which both
        // keep the type tag; the bare ISO string dropped the type.
        GVal::Temporal(t) => out.push_str(&t.json_tagged()),
        // An element serializes to the full `{id, labels, properties}` (edge:
        // `{id, from, to, labels, properties}`) form — byte-identical to GQL and the
        // TS engine, via the shared canonical `Value::Map`.
        GVal::Node(i) => push_result_value(out, &crate::gql::eval::node_result_value(graph, *i)),
        GVal::Edge(i) => push_result_value(out, &crate::gql::eval::edge_result_value(graph, *i)),
        GVal::List(items) => {
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_gval(out, graph, x);
            }
            out.push(']');
        }
        GVal::Map(entries) => {
            // Match serde_json::Map (a BTreeMap): keys sorted lexicographically.
            // The sync live-query layer diffs cells by `JSON.stringify`
            // byte-equality, so this canonical order is load-bearing.
            let mut pairs: Vec<(String, &GVal)> = entries
                .iter()
                .map(|(k, val)| (map_key(graph, k), val))
                .collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            out.push('{');
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_json_str(out, k);
                out.push(':');
                write_gval(out, graph, val);
            }
            out.push('}');
        }
        // A property element serializes as `{"key":…,"value":…}` (TinkerPop's
        // property shape); its owner back-reference is internal and not emitted.
        GVal::Property(p) => {
            out.push_str("{\"key\":");
            push_json_str(out, &p.key);
            out.push_str(",\"value\":");
            write_gval(out, graph, &p.value);
            out.push('}');
        }
        // GQL-only (see `crate::value`). Unreachable from a traversal, and
        // rendering `null` keeps the writer total rather than panicking if some
        // future step ever does carry one.
        GVal::Record(_) | GVal::Path(_) => out.push_str("null"),
    }
}

/// A map key's string form: a `Str` verbatim, anything else via its JSON text
/// (a number through `js_number`, etc.) — matching serde's key stringification.
fn map_key(graph: &Graph, k: &GVal) -> String {
    match k {
        GVal::Str(s) => s.to_string(),
        other => {
            let mut tmp = String::new();
            write_gval(&mut tmp, graph, other);
            tmp
        }
    }
}

fn run_steps(graph: &mut Graph, ctx: &mut Ctx, steps: &[Step], mut stream: Vec<Trav>) -> Vec<Trav> {
    for step in steps {
        stream = apply(graph, ctx, step, stream);
    }
    stream
}

/// Run a sub-plan from a single seed value; collect its output values.
fn sub_vals(graph: &mut Graph, ctx: &mut Ctx, plan: &Traversal, seed: &Trav) -> Vec<GVal> {
    run_steps(graph, ctx, &plan.steps, vec![seed.clone()])
        .into_iter()
        .map(|t| t.val)
        .collect()
}

fn sub_nonempty(graph: &mut Graph, ctx: &mut Ctx, plan: &Traversal, seed: &Trav) -> bool {
    !run_steps(graph, ctx, &plan.steps, vec![seed.clone()]).is_empty()
}

// --- match() solver ---------------------------------------------------------
//
// Port of the TS `executor/match.ts`. Each pattern is `as(start) … [as(end)]`:
// from the value bound to `start`, run the inner traversal, then bind `end` to
// the output (if unbound) or filter against it (if bound). No trailing `as` ⇒ a
// pure filter on `start`; a `not(...)`/`where(...)` wrapper ⇒ a (negated) filter.
// `GVal` already compares graph elements by their dense id, so `==` is identity.

struct MatchPattern {
    start_key: String,
    end_key: Option<String>,
    inner: Traversal,
    negated: bool,
}

/// Lower one pattern plan into a {@link MatchPattern}.
fn parse_pattern(plan: &Traversal) -> MatchPattern {
    let steps = &plan.steps;
    // `not(inner)` / `where(inner)` filter wrappers (single step): parse the inner
    // pattern and flip negation (`where` keeps it positive).
    if steps.len() == 1 {
        if let Step::Not(inner) = &steps[0] {
            let mut p = parse_pattern(inner);
            p.negated = !p.negated;
            return p;
        }
        if let Step::Where(inner) = &steps[0] {
            return parse_pattern(inner);
        }
    }
    let start_key = match steps.first() {
        Some(Step::As(l)) => l.clone(),
        // Malformed (no leading as): an unbindable start ⇒ the pattern never runs.
        _ => String::new(),
    };
    if steps.len() >= 2 {
        if let Some(Step::As(end)) = steps.last() {
            let inner = Traversal {
                steps: steps[1..steps.len() - 1].to_vec(),
            };
            return MatchPattern {
                start_key,
                end_key: Some(end.clone()),
                inner,
                negated: false,
            };
        }
    }
    let inner = Traversal {
        steps: steps.get(1..).unwrap_or(&[]).to_vec(),
    };
    MatchPattern {
        start_key,
        end_key: None,
        inner,
        negated: false,
    }
}

/// The seed label: a pattern *start* that is never a binding *end* (a source).
fn match_start_label(patterns: &[MatchPattern]) -> String {
    let ends: Vec<&String> = patterns
        .iter()
        .filter(|p| !p.negated)
        .filter_map(|p| p.end_key.as_ref())
        .collect();
    for p in patterns {
        if !ends.iter().any(|e| **e == p.start_key) {
            return p.start_key.clone();
        }
    }
    patterns
        .first()
        .map(|p| p.start_key.clone())
        .unwrap_or_default()
}

/// `t` with `key` bound to a single `val` (match binds each label once).
fn match_bind(t: &Trav, key: &str, val: GVal) -> Trav {
    let mut nt = t.clone();
    let tags = Arc::make_mut(&mut nt.tags);
    match tags.iter_mut().find(|(l, _)| l == key) {
        Some((_, list)) => *list = vec![val],
        None => tags.push((key.to_string(), vec![val])),
    }
    nt
}

/// Apply one pattern to a traverser, returning the consistent continuations.
fn apply_pattern(graph: &mut Graph, ctx: &mut Ctx, p: &MatchPattern, t: &Trav) -> Vec<Trav> {
    let Some(start_val) = t.recall(&p.start_key, Pop::Last) else {
        return vec![];
    };
    let seed = Trav {
        val: start_val,
        path: t.path.clone(),
        tags: t.tags.clone(),
        loops: t.loops,
        sack: t.sack.clone(),
    };
    let outs = sub_vals(graph, ctx, &p.inner, &seed);
    let bound_end = p.end_key.as_ref().and_then(|k| t.recall(k, Pop::Last));

    if p.negated {
        let satisfiable = outs
            .iter()
            .any(|o| bound_end.as_ref().is_none_or(|b| o == b));
        return if satisfiable { vec![] } else { vec![t.clone()] };
    }
    let Some(end_key) = &p.end_key else {
        return if outs.is_empty() {
            vec![]
        } else {
            vec![t.clone()]
        }; // pure filter
    };
    if let Some(b) = bound_end {
        return if outs.contains(&b) {
            vec![t.clone()]
        } else {
            vec![]
        };
    }
    // Bind the end label, one branch per distinct candidate value. Dedup via the
    // same hashable key as `Step::Dedupe` (O(n), not the old O(n²) linear scan);
    // a NaN candidate has no key, so it's never a duplicate and always branches.
    let mut seen: HashSet<DedupKey> = HashSet::new();
    let mut branches = Vec::new();
    for o in outs {
        match dedup_key(&o) {
            None => branches.push(match_bind(t, end_key, o)),
            Some(k) => {
                if seen.insert(k) {
                    branches.push(match_bind(t, end_key, o));
                }
            }
        }
    }
    branches
}

/// Pick a not-yet-applied pattern whose start is bound, preferring binders.
fn pick_runnable(patterns: &[MatchPattern], done: &[bool], t: &Trav) -> Option<usize> {
    let mut negated = None;
    for (i, p) in patterns.iter().enumerate() {
        if done[i] || t.recall(&p.start_key, Pop::Last).is_none() {
            continue;
        }
        if !p.negated {
            return Some(i);
        }
        if negated.is_none() {
            negated = Some(i);
        }
    }
    negated
}

/// Depth-first join: apply runnable patterns until all are satisfied, emitting
/// one traverser (carrying the binding tags) per consistent assignment.
fn match_solve(
    graph: &mut Graph,
    ctx: &mut Ctx,
    patterns: &[MatchPattern],
    t: Trav,
    done: &mut Vec<bool>,
    out: &mut Vec<Trav>,
) {
    if done.iter().all(|&d| d) {
        // Emit the binding map as the value (TinkerPop-faithful); tags carry the
        // bindings for any following select(...).
        let bindings: Vec<(GVal, GVal)> = t
            .tags
            .iter()
            .filter_map(|(l, vs)| {
                vs.last()
                    .map(|v| (GVal::Str(Arc::from(l.as_str())), v.clone()))
            })
            .collect();
        out.push(t.with(GVal::Map(bindings)));
        return;
    }
    let Some(idx) = pick_runnable(patterns, done, &t) else {
        return; // stuck: this branch contributes nothing
    };
    done[idx] = true;
    for t2 in apply_pattern(graph, ctx, &patterns[idx], &t) {
        match_solve(graph, ctx, patterns, t2, done, out);
    }
    done[idx] = false; // backtrack
}

// --- shortestPath() solver --------------------------------------------------
//
// Port of the TS executor/shortest-path.ts: unweighted BFS over incident edges
// (both directions), emitting all shortest vertex paths from each source.

/// All shortest (fewest-hop) vertex paths from `src` to each destination, as
/// vertex-index arrays `[src, …, dest]`. `targets` (None ⇒ every reached vertex)
/// filters destinations; equal-length alternatives are all returned.
fn shortest_paths_from(
    graph: &Graph,
    src: u32,
    targets: Option<&HashSet<u32>>,
    out: bool,
    inn: bool,
) -> Vec<Vec<u32>> {
    let mut dist: HashMap<u32, usize> = HashMap::from([(src, 0)]);
    let mut preds: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut frontier = vec![src];
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for &v in &frontier {
            let d = dist[&v];
            for n in crate::seek::adj(
                graph,
                v,
                seek_dir(out, inn),
                &[],
                crate::seek::SelfLoops::Twice,
            )
            .map(|a| a.nbr)
            {
                match dist.get(&n).copied() {
                    None => {
                        dist.insert(n, d + 1);
                        preds.insert(n, vec![v]);
                        next.push(n);
                    }
                    Some(nd) if nd == d + 1 => preds.entry(n).or_default().push(v),
                    _ => {}
                }
            }
        }
        frontier = next;
    }
    let mut paths = Vec::new();
    for &id in dist.keys() {
        if targets.is_none_or(|t| t.contains(&id)) {
            build_paths(src, id, &[], &preds, &mut paths);
        }
    }
    paths
}

/// Reconstruct every shortest path to `id` by walking predecessors back to `src`.
fn build_paths(
    src: u32,
    id: u32,
    tail: &[u32],
    preds: &HashMap<u32, Vec<u32>>,
    out: &mut Vec<Vec<u32>>,
) {
    let mut path = vec![id];
    path.extend_from_slice(tail);
    if id == src {
        out.push(path);
        return;
    }
    for &p in preds.get(&id).map(Vec::as_slice).unwrap_or_default() {
        build_paths(src, p, &path, preds, out);
    }
}

/// Run an in-process OLAP algorithm (`pageRank`/`connectedComponent`/`peerPressure`)
/// over the whole graph, writing each vertex's result to `property`. The computation
/// is local — `withComputer()` is accepted only as a spec-currency marker. The
/// incoming traversers pass straight through so downstream steps read the written
/// property (`g.V().pageRank().order().by('...')`).
fn algo_step(
    graph: &mut Graph,
    ctx: &mut Ctx,
    stream: Vec<Trav>,
    name: &str,
    property: String,
    configure: impl FnOnce(&mut crate::algo::AlgoConfig),
) -> Vec<Trav> {
    let mut cfg = crate::algo::AlgoConfig {
        write_property: Some(property),
        ..Default::default()
    };
    configure(&mut cfg);
    if crate::algo::run_with(graph, name, &cfg).is_err() {
        ctx.fault.get_or_insert((
            crate::error_codes::ErrorCode::InvalidValue,
            "graph algorithm step failed".into(),
        ));
        return Vec::new();
    }
    stream
}

fn shortest_path_step(
    graph: &mut Graph,
    ctx: &mut Ctx,
    target: Option<&Traversal>,
    out: bool,
    inn: bool,
    stream: Vec<Trav>,
) -> Vec<Trav> {
    // Resolve the destination set once: run the target sub-plan over every vertex.
    // Collect the indices first so the immutable borrow is released before the
    // mutable `run_steps` call inside the filter.
    let targets: Option<HashSet<u32>> = target.map(|plan| {
        let verts: Vec<u32> = graph.vertex_indices().collect();
        verts
            .into_iter()
            .filter(|&v| {
                !run_steps(graph, ctx, &plan.steps, vec![Trav::root(GVal::Node(v))]).is_empty()
            })
            .collect()
    });
    let mut next = Vec::new();
    for t in &stream {
        if let GVal::Node(src) = t.val {
            for path in shortest_paths_from(graph, src, targets.as_ref(), out, inn) {
                next.push(t.with(GVal::List(path.into_iter().map(GVal::Node).collect())));
            }
        }
    }
    next
}

fn match_step(
    graph: &mut Graph,
    ctx: &mut Ctx,
    plans: &[Traversal],
    stream: Vec<Trav>,
) -> Vec<Trav> {
    let patterns: Vec<MatchPattern> = plans.iter().map(parse_pattern).collect();
    let start_label = match_start_label(&patterns);
    let mut out = Vec::new();
    for t in stream {
        // Seed the source label from the incoming value unless already bound.
        let seed = if t.recall(&start_label, Pop::Last).is_some() {
            t
        } else {
            let v = t.val.clone();
            match_bind(&t, &start_label, v)
        };
        let mut done = vec![false; patterns.len()];
        match_solve(graph, ctx, &patterns, seed, &mut done, &mut out);
    }
    out
}

// --- value helpers ----------------------------------------------------------

fn value_to_gval(v: Value) -> GVal {
    GVal::from_stored(&v, false)
}

fn gval_to_value(v: &GVal) -> Value {
    match v {
        GVal::Null => Value::Null,
        GVal::Bool(b) => Value::Bool(*b),
        GVal::Num(n) => Value::Num(*n),
        GVal::Str(s) => Value::Str(s.clone()),
        GVal::Temporal(t) => Value::Temporal(*t),
        GVal::List(items) => Value::List(items.iter().map(gval_to_value).collect()),
        // A Gremlin map written back to a property → a stored record. Stored map
        // keys are strings; a scalar key coerces to its string form, any richer
        // key (element/list/map) can't be a field name and is dropped. The store
        // canonicalizes (sorts) on write.
        GVal::Map(pairs) => Value::Map(
            pairs
                .iter()
                .filter_map(|(k, v)| {
                    let key: std::sync::Arc<str> = match k {
                        GVal::Str(s) => s.clone(),
                        GVal::Num(n) => crate::jsonfmt::js_number(*n).into(),
                        GVal::Bool(b) => if *b { "true" } else { "false" }.into(),
                        GVal::Null => "null".into(),
                        _ => return None,
                    };
                    Some((key, gval_to_value(v)))
                })
                .collect(),
        ),
        _ => Value::Null,
    }
}

fn prop(graph: &Graph, v: &GVal, key: &str) -> GVal {
    match v {
        GVal::Node(i) => value_to_gval(graph.props.value(*i as usize, key, &graph.strs)),
        GVal::Edge(e) => value_to_gval(graph.edge_props.value(*e as usize, key, &graph.strs)),
        _ => GVal::Null,
    }
}

/// A `{ key: value }` map of an element's present properties (a stored null is
/// present and rides through as a `Null` value).
fn element_props_map(graph: &Graph, v: &GVal) -> GVal {
    // `present_keys` is already presence-gated, so include every present key —
    // a present null rides through as a `Null` value (not dropped).
    let entries: Vec<(GVal, GVal)> = present_keys(graph, v)
        .into_iter()
        .map(|k| (GVal::Str(Arc::from(k.as_str())), prop(graph, v, &k)))
        .collect();
    GVal::Map(entries)
}

/// A self-describing vertex record for a subgraph cap: `{ id, labels, properties }`.
fn subgraph_vertex(graph: &Graph, v: u32) -> GVal {
    let gv = GVal::Node(v);
    let labels: Vec<GVal> = graph
        .vertex_labels(v)
        .iter()
        .map(|&l| GVal::Str(graph.labels.arc(l)))
        .collect();
    GVal::Map(vec![
        (GVal::Str(Arc::from("id")), GVal::Str(graph.vid.arc(v))),
        (GVal::Str(Arc::from("labels")), GVal::list(labels)),
        (
            GVal::Str(Arc::from("properties")),
            element_props_map(graph, &gv),
        ),
    ])
}

/// A self-describing edge record: `{ id, label, outV, inV, properties }`.
fn subgraph_edge(graph: &Graph, e: u32) -> GVal {
    let ge = GVal::Edge(e);
    let outv = GVal::Node(graph.e_src[e as usize]);
    let inv = GVal::Node(graph.e_dst[e as usize]);
    GVal::Map(vec![
        (GVal::Str(Arc::from("id")), elem_id(graph, &ge)),
        (
            GVal::Str(Arc::from("label")),
            GVal::Str(graph.etype.arc(graph.e_type[e as usize])),
        ),
        (GVal::Str(Arc::from("outV")), elem_id(graph, &outv)),
        (GVal::Str(Arc::from("inV")), elem_id(graph, &inv)),
        (
            GVal::Str(Arc::from("properties")),
            element_props_map(graph, &ge),
        ),
    ])
}

/// Every present property of `v` as `(column id, name)`.
///
/// `present_keys` walks the store BY ID and then throws the id away, returning
/// names — so every caller hashed each name straight back to the id it had just
/// discarded, twice over (`prop_present`, then `prop`). This keeps it.
///
/// Measured NEUTRAL on a no-argument `valueMap()` over 50k x 12 properties (79.6
/// ms before, 79.2-82.6 after). It is here because deriving a value, discarding
/// it and re-deriving it twice is worth removing on its own; the timing says only
/// that `valueMap`'s cost is elsewhere — building a `GVal::Map`, an `Arc` key and
/// a boxed value per entry, 600k of them. That is the thing to optimize next, and
/// it is why this is not the win it looks like.
fn present_key_ids(graph: &Graph, v: &GVal) -> Vec<(u32, Arc<str>)> {
    if let Some(GVal::Str(k)) = prop_key_field(v) {
        return vec![(u32::MAX, k)]; // a property element exposes only its own key
    }

    let (store, idx) = match v {
        GVal::Node(i) => (&graph.props, *i as usize),
        GVal::Edge(e) => (&graph.edge_props, *e as usize),
        _ => return Vec::new(),
    };

    (0..store.keys.len() as u32)
        // Presence, not value: a stored null is a present property (first-class
        // value — divergence from TinkerPop, which has no null property values).
        .filter(|&kid| store.is_present_id(idx, kid))
        .map(|kid| (kid, Arc::from(store.keys.text(kid))))
        .collect()
}

/// The value of the already-resolved column `kid` on `v`. See
/// [`present_key_ids`]; `u32::MAX` means "a property element's own value".
fn prop_by_id(graph: &Graph, v: &GVal, kid: u32) -> GVal {
    if kid == u32::MAX {
        return prop_value_field(v).unwrap_or(GVal::Null);
    }

    match v {
        GVal::Node(i) => value_to_gval(graph.props.value_id(*i as usize, kid, &graph.strs)),
        GVal::Edge(e) => value_to_gval(graph.edge_props.value_id(*e as usize, kid, &graph.strs)),
        _ => GVal::Null,
    }
}

fn present_keys(graph: &Graph, v: &GVal) -> Vec<String> {
    // A property element (from `properties()`) exposes its own single key, so
    // `hasKey`/`hasNot` filter a property stream by the property's key field.
    if let Some(GVal::Str(k)) = prop_key_field(v) {
        return vec![k.to_string()];
    }
    let (store, idx) = match v {
        GVal::Node(i) => (&graph.props, *i as usize),
        GVal::Edge(e) => (&graph.edge_props, *e as usize),
        _ => return Vec::new(),
    };
    (0..store.keys.len() as u32)
        // Presence, not value: a stored null is a present property (first-class
        // value — divergence from TinkerPop, which has no null property values).
        .filter(|&kid| store.is_present_id(idx, kid))
        .map(|kid| store.keys.text(kid).to_string())
        .collect()
}

/// Is property `key` present on element `v`? A stored null counts as present, so
/// projection steps gate inclusion on this (not `prop(...) != Null`, which also
/// drops a present null). Property elements / non-elements: not applicable.
fn prop_present(graph: &Graph, v: &GVal, key: &str) -> bool {
    match v {
        GVal::Node(i) => graph.props.is_present(*i as usize, key),
        GVal::Edge(e) => graph.edge_props.is_present(*e as usize, key),
        _ => false,
    }
}

fn elem_id(graph: &Graph, v: &GVal) -> GVal {
    match v {
        GVal::Node(i) => GVal::Str(graph.vid.arc(*i)),
        // Every edge has an id (assigned external id, else canonical `e{index}`).
        GVal::Edge(e) => GVal::Str(Arc::from(graph.edge_id(*e).as_ref())),
        // A non-element has no id → NULL, exactly as its sibling `elem_label`
        // does and as the TS engine does. This used to pass the value THROUGH,
        // so `path().id()` handed the paths back untouched instead of nulls, and
        // a following `sum()` then faulted on them where TS summed nulls. Found
        // by the Gremlin differential fuzzer.
        _ => GVal::Null,
    }
}

fn elem_label(graph: &Graph, v: &GVal) -> GVal {
    match v {
        GVal::Node(i) => match graph.vertex_labels(*i).first() {
            Some(&lid) => GVal::Str(graph.labels.arc(lid)),
            None => GVal::Null,
        },
        GVal::Edge(e) => GVal::Str(graph.etype.arc(graph.e_type[*e as usize])),
        _ => GVal::Null,
    }
}

/// A number for a numeric aggregate (`sum`/`mean`). `null` is skipped by the
/// caller; any other non-number is a type error (flagged), matching TinkerPop —
/// rather than silently coercing strings/bools to numbers.
fn strict_num(v: &GVal) -> Option<f64> {
    match v {
        GVal::Num(n) => Some(*n),
        GVal::Null => None,
        _ => {
            set_type_fault();
            None
        }
    }
}

/// The distinct identifiers in a `math()` expression, in first-appearance
/// order. Used to map `by()` modulators to operands (TinkerPop cycles them in
/// this order). Mirrors the TS `mathVars`.
fn math_vars(expr: &str) -> Vec<String> {
    let chars: Vec<char> = expr.chars().collect();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            // An identifier immediately followed by `(` (whitespace allowed) is a
            // function call (`sin`, `atan2`, …), not an operand — skip it so it
            // neither consumes a by() modulator nor is looked up as an unbound
            // tag. Mirrors the TS `mathVars`.
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && chars[j] == '(' {
                continue;
            }
            let name: String = chars[start..i].iter().collect();
            if seen.insert(name.clone()) {
                out.push(name);
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Numeric constant identifiers recognized in `math()` (mXparser's `pi`/`e`).
/// Only used when the name is not shadowed by a bound variable — the parser
/// checks `vals` first. Mirrors the TS `MATH_CONSTS`.
fn math_const(name: &str) -> Option<f64> {
    match name {
        "pi" => Some(std::f64::consts::PI),
        "e" => Some(std::f64::consts::E),
        _ => None,
    }
}

/// `math()` `signum`: -1 | 0 | 1 with NaN passing through. Matches the GQL
/// `sign` kernel (NOT `f64::signum`, which yields +1 for 0.0) and the TS twin.
fn math_signum(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The unary `math()` functions, keyed by name. Each is the SAME f64 primitive
/// the GQL `call_scalar` kernel uses (`f64::sin`, …), so `math()` stays
/// bit-identical to GQL and to the TS twin. Membership also decides the
/// bare/juxtaposition form (`sin _` == `sin(_)`) — only unary functions take it.
fn unary_math_fn(name: &str) -> Option<fn(f64) -> f64> {
    Some(match name {
        "sin" => f64::sin,
        "cos" => f64::cos,
        "tan" => f64::tan,
        "asin" => f64::asin,
        "acos" => f64::acos,
        "atan" => f64::atan,
        "sinh" => f64::sinh,
        "cosh" => f64::cosh,
        "tanh" => f64::tanh,
        "sqrt" => f64::sqrt,
        "abs" => f64::abs,
        "ceil" => f64::ceil,
        "floor" => f64::floor,
        "exp" => f64::exp,
        "ln" => f64::ln,
        "log10" => f64::log10,
        "signum" => math_signum,
        _ => return None,
    })
}

/// Dispatch a `math()` function call. `b` is `Some` for the 2-arg forms
/// (`atan2`/`pow`/`log`), which REQUIRE parens — the bare form is unary-only.
/// `pow` inherits GQL's `Power`: `x.powf(y)` matches JS `x ** y` except for a
/// ≤1-ULP glibc-`powf`-vs-V8-`pow` difference on some inputs (a documented
/// won't-fix). Arity mismatch
/// → `None` (fault). Note: `log(base, value)` and `ln` (natural) follow GQL
/// naming; TinkerPop/mXparser spells natural log `log`.
fn math_call(name: &str, a: f64, b: Option<f64>) -> Option<f64> {
    if let Some(y) = b {
        return match name {
            "atan2" => Some(a.atan2(y)),
            "pow" => Some(a.powf(y)),
            "log" => Some(y.ln() / a.ln()),
            _ => None,
        };
    }
    unary_math_fn(name).map(|f| f(a))
}

/// Recursive-descent evaluator for the `math()` grammar. Precedence, loosest to
/// tightest (mXparser / TinkerPop): `+ -` < `* / %` < `^` (right-assoc) < unary
/// `- +` < primary. Primary = numeric literal, parenthesized expr, `name(args)`
/// function call, bare/juxtaposition unary application (`sin _`), constant
/// (`pi`/`e`), or an identifier resolved via `vals` (variables win over
/// constants and function names). Returns `None` on any malformed expression or
/// unknown identifier — surfaced as an `InvalidValue` fault, the SAME code the
/// TS twin raises. A faithful port of the TS `evalMath`.
struct MathP<'a> {
    c: Vec<char>,
    pos: usize,
    vals: &'a HashMap<String, f64>,
}

impl MathP<'_> {
    fn peek(&self) -> char {
        self.c.get(self.pos).copied().unwrap_or('\0')
    }
    fn skip(&mut self) {
        while self.peek().is_whitespace() {
            self.pos += 1;
        }
    }
    fn primary(&mut self) -> Option<f64> {
        self.skip();
        let ch = self.peek();
        if ch == '(' {
            self.pos += 1;
            let v = self.add()?;
            self.skip();
            if self.peek() != ')' {
                return None;
            }
            self.pos += 1;
            return Some(v);
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = self.pos;
            while self.peek().is_ascii_alphanumeric() || self.peek() == '_' {
                self.pos += 1;
            }
            let name: String = self.c[start..self.pos].iter().collect();
            // Variables win over constants and function names: resolve `vals`
            // first (the eager loop pre-populated it for bound tags / `_`).
            if let Some(v) = self.vals.get(&name) {
                return Some(*v);
            }
            // Function call: identifier immediately followed by `(`.
            self.skip();
            if self.peek() == '(' {
                self.pos += 1; // consume '('
                let a = self.add()?;
                self.skip();
                let mut b = None;
                if self.peek() == ',' {
                    self.pos += 1;
                    b = Some(self.add()?);
                    self.skip();
                }
                if self.peek() != ')' {
                    return None;
                }
                self.pos += 1;
                return math_call(&name, a, b);
            }
            // Bare/juxtaposition form (TinkerPop): a unary function name NOT
            // followed by `(` applies to the next unary expression. Binds tighter
            // than binary ops (`sin _ + 1` == `(sin _) + 1`) and chains
            // right-associatively (`sin cos _` == `sin(cos(_))`); the unary arg
            // also allows a leading sign (`abs -3` == `abs(-3)`). Multi-arg
            // functions still require parens (handled above), so they fall
            // through here to a fault.
            if let Some(f) = unary_math_fn(&name) {
                let arg = self.unary()?;
                return Some(f(arg));
            }
            // Unshadowed constant (`pi`/`e`), else an unbound identifier (fault).
            return math_const(&name);
        }
        // Numeric literal (a leading sign is handled by `unary`).
        let start = self.pos;
        while self.peek().is_ascii_digit() || self.peek() == '.' {
            self.pos += 1;
        }
        if start == self.pos {
            return None;
        }
        let lit: String = self.c[start..self.pos].iter().collect();
        lit.parse::<f64>().ok()
    }
    fn power(&mut self) -> Option<f64> {
        // Unary binds tighter than `^` (mXparser): `-2 ^ 2` == `(-2) ^ 2` == 4.
        let base = self.unary()?;
        self.skip();
        if self.peek() == '^' {
            self.pos += 1;
            // Right-associative: `2 ^ 3 ^ 2` == `2 ^ (3 ^ 2)` == 512.
            let exp = self.power()?;
            return Some(base.powf(exp));
        }
        Some(base)
    }
    fn unary(&mut self) -> Option<f64> {
        self.skip();
        match self.peek() {
            '-' => {
                self.pos += 1;
                Some(-self.unary()?)
            }
            '+' => {
                self.pos += 1;
                self.unary()
            }
            _ => self.primary(),
        }
    }
    fn mul(&mut self) -> Option<f64> {
        let mut left = self.power()?;
        self.skip();
        while matches!(self.peek(), '*' | '/' | '%') {
            let op = self.peek();
            self.pos += 1;
            let right = self.power()?;
            left = match op {
                '*' => left * right,
                '/' => left / right,
                _ => left % right,
            };
            self.skip();
        }
        Some(left)
    }
    fn add(&mut self) -> Option<f64> {
        let mut left = self.mul()?;
        self.skip();
        while self.peek() == '+' || self.peek() == '-' {
            let op = self.peek();
            self.pos += 1;
            let right = self.mul()?;
            left = if op == '+' {
                left + right
            } else {
                left - right
            };
            self.skip();
        }
        Some(left)
    }
}

fn eval_math_expr(expr: &str, vals: &HashMap<String, f64>) -> Option<f64> {
    let mut p = MathP {
        c: expr.chars().collect(),
        pos: 0,
        vals,
    };
    let r = p.add()?;
    p.skip();
    if p.pos < p.c.len() {
        return None; // trailing input
    }
    Some(r)
}

/// Order two values the way TinkerPop's `Comparable` does: numbers with numbers,
/// strings with strings, booleans with booleans. No string/bool→number coercion.
/// `None` ⇒ not comparable (different types, an element, or a null operand).
fn gcmp(a: &GVal, b: &GVal) -> Option<Ordering> {
    match (a, b) {
        (GVal::Num(x), GVal::Num(y)) => x.partial_cmp(y),
        (GVal::Str(x), GVal::Str(y)) => Some(x.as_ref().cmp(y.as_ref())),
        (GVal::Bool(x), GVal::Bool(y)) => Some(x.cmp(y)),
        // Temporals of the same instant kind order chronologically; durations and
        // cross-kind pairs are not relationally ordered (`rel_cmp` → None) and so
        // fault as incomparable, matching the GQL relational policy and the TS
        // Gremlin `compareValues`. Without this arm an as-of predicate —
        // `has('vf', lte(date('2021-06-01')))` — raised E_INVALID_VALUE even once
        // the grammar could express the literal, so the whole bitemporal query
        // shape was unwritable on this dialect.
        (GVal::Temporal(x), GVal::Temporal(y)) => x.rel_cmp(y),
        _ => None,
    }
}

/// `gcmp`, but flag a type fault when two genuinely incomparable *non-null*
/// values are ordered (TinkerPop's `ClassCastException`). A null/missing operand
/// is simply not comparable and filtered out — no fault — so `has(k, gt(n))` on
/// a missing key still just drops the traverser.
fn cmp_or_fault(a: &GVal, b: &GVal) -> Option<Ordering> {
    let c = gcmp(a, b);
    if c.is_none() && !matches!(a, GVal::Null) && !matches!(b, GVal::Null) {
        set_type_fault();
    }
    c
}

fn gval_type_rank(v: &GVal) -> u8 {
    match v {
        GVal::Null => 0,
        GVal::Bool(_) => 1,
        GVal::Num(_) => 2,
        GVal::Str(_) => 3,
        GVal::Temporal(_) => 4,
        GVal::Node(_) => 5,
        GVal::Edge(_) => 6,
        GVal::List(_) => 7,
        GVal::Map(_) => 8,
        GVal::Property(_) => 9,
        // GQL-only variants (`Val` and `GVal` are one type — see
        // `crate::value`). A traversal cannot produce them, but the order must
        // stay TOTAL for every inhabitant of the type or `sort_by` can panic,
        // and under `panic = "abort"` that takes the host down. Ranking them
        // past everything Gremlin can make is both cheap and safe.
        GVal::Record(_) => 10,
        GVal::Path(_) => 11,
    }
}

/// A genuine TOTAL order over `GVal`, used only as the `order()` sort comparator's
/// tie-break so `slice::sort_by` can never see a non-total order — Rust panics on
/// that ("comparison function does not implement a total order"), which under the
/// release `panic = "abort"` build would abort the host on ordinary mixed-type
/// input. It AGREES with `gcmp` on every comparable pair (numbers by value, same-
/// kind temporals chronologically, …), so a homogeneous stream sorts identically;
/// for incomparable pairs (cross-type, NaN, cross-kind temporal) it falls back to a
/// deterministic order (type-rank, then a within-type total order — `f64::total_cmp`
/// puts NaN last). `order()` still records a type fault for those pairs via
/// `cmp_or_fault`, so `try_run` surfaces the error and this tie-break order is never
/// observed there; it only makes the infallible `run` path deterministic.
fn gcmp_total(a: &GVal, b: &GVal) -> Ordering {
    let (ra, rb) = (gval_type_rank(a), gval_type_rank(b));
    if ra != rb {
        return ra.cmp(&rb);
    }
    match (a, b) {
        (GVal::Bool(x), GVal::Bool(y)) => x.cmp(y),
        (GVal::Num(x), GVal::Num(y)) => x.total_cmp(y),
        (GVal::Str(x), GVal::Str(y)) => x.as_ref().cmp(y.as_ref()),
        (GVal::Temporal(x), GVal::Temporal(y)) => x.cmp_total(y),
        (GVal::Node(x), GVal::Node(y)) | (GVal::Edge(x), GVal::Edge(y)) => x.cmp(y),
        (GVal::List(x), GVal::List(y)) => x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| gcmp_total(xi, yi))
            .find(|o| *o != Ordering::Equal)
            .unwrap_or_else(|| x.len().cmp(&y.len())),
        (GVal::Map(x), GVal::Map(y)) => x
            .iter()
            .zip(y.iter())
            .map(|((k1, v1), (k2, v2))| gcmp_total(k1, k2).then_with(|| gcmp_total(v1, v2)))
            .find(|o| *o != Ordering::Equal)
            .unwrap_or_else(|| x.len().cmp(&y.len())),
        (GVal::Property(a), GVal::Property(b)) => a
            .key
            .as_ref()
            .cmp(b.key.as_ref())
            .then_with(|| gcmp_total(&a.value, &b.value)),
        // Same rank ⟹ same variant (Null==Null falls here as Equal).
        _ => Ordering::Equal,
    }
}

thread_local! {
    /// Compile each `regex()` pattern once and reuse it per value — the pattern
    /// is re-applied across the whole stream. Mirrors the TS `regexCache`.
    static REGEX_CACHE: RefCell<HashMap<String, regex::Regex>> = RefCell::new(HashMap::new());
}

/// Whether `hay` matches the (already-parse-validated) pattern `pat`. Like JS
/// `RegExp.test` / TinkerPop `TextP.regex`, the match is unanchored (searches
/// anywhere). Compilation is cached; a bad pattern (shouldn't occur post-parse)
/// simply doesn't match.
fn regex_is_match(pat: &str, hay: &str) -> bool {
    REGEX_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if !cache.contains_key(pat) {
            if cache.len() >= 1000 {
                cache.clear(); // bound memory; patterns are typically few
            }
            match regex::Regex::new(pat) {
                Ok(re) => {
                    cache.insert(pat.to_string(), re);
                }
                Err(_) => return false,
            }
        }
        cache.get(pat).is_some_and(|re| re.is_match(hay))
    })
}

fn p_matches(p: &P, v: &GVal) -> bool {
    let cmp = |t: &GVal, want: Ordering| cmp_or_fault(v, t) == Some(want);
    let ge = |t: &GVal| {
        matches!(
            cmp_or_fault(v, t),
            Some(Ordering::Greater | Ordering::Equal)
        )
    };
    let le = |t: &GVal| matches!(cmp_or_fault(v, t), Some(Ordering::Less | Ordering::Equal));
    let s = |g: &GVal| match g {
        GVal::Str(s) => Some(s.to_string()),
        _ => None,
    };
    match p {
        P::Eq(t) => v == t,
        P::Neq(t) => v != t,
        P::Gt(t) => cmp(t, Ordering::Greater),
        P::Lt(t) => cmp(t, Ordering::Less),
        P::Gte(t) => ge(t),
        P::Lte(t) => le(t),
        P::Between(lo, hi) => ge(lo) && cmp(hi, Ordering::Less),
        P::Inside(lo, hi) => cmp(lo, Ordering::Greater) && cmp(hi, Ordering::Less),
        P::Outside(lo, hi) => cmp(lo, Ordering::Less) || cmp(hi, Ordering::Greater),
        P::Within(vs) => vs.contains(v),
        P::Without(vs) => !vs.contains(v),
        P::StartsWith(p) => s(v).is_some_and(|x| x.starts_with(p)),
        P::EndingWith(p) => s(v).is_some_and(|x| x.ends_with(p)),
        P::Containing(p) => s(v).is_some_and(|x| x.contains(p)),
        P::NotContaining(p) => s(v).is_some_and(|x| !x.contains(p)),
        P::Regex(pat) => s(v).is_some_and(|x| regex_is_match(pat, &x)),
        P::Not(inner) => !p_matches(inner, v),
    }
}

fn token_project(graph: &Graph, tok: Token, v: &GVal) -> GVal {
    match tok {
        Token::Id => elem_id(graph, v),
        Token::Label => elem_label(graph, v),
        Token::Key | Token::Value => match v {
            // Project a {key, value} property map.
            GVal::Map(entries) => {
                let want = if tok == Token::Key { "key" } else { "value" };
                entries
                    .iter()
                    .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == want))
                    .map(|(_, x)| x.clone())
                    .unwrap_or(GVal::Null)
            }
            _ => GVal::Null,
        },
    }
}

/// Resolve a `by()` modulator against `value`.
fn eval_by(graph: &mut Graph, ctx: &mut Ctx, by: &By, value: &GVal) -> GVal {
    match by {
        By::Identity(_) => value.clone(),
        By::Key(key, _) => match value {
            GVal::Node(_) | GVal::Edge(_) => prop(graph, value, key),
            // `by('k')` over a Map (`group`/`groupCount`/`project()` row) projects
            // the value at that key — e.g. `project('name','age').order().by('age')`.
            // Without this the whole Map reached the comparator ("cannot order …").
            GVal::Map(entries) => entries
                .iter()
                .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == key.as_str()))
                .map(|(_, v)| v.clone())
                .unwrap_or(GVal::Null),
            _ => value.clone(),
        },
        By::Token(tok, _) => token_project(graph, *tok, value),
        // `Column.keys` / `Column.values` over a Map yields the list of its keys /
        // values (TinkerPop `select(Column)`); `order(local)` special-cases it to
        // sort a Map's entries and never routes through here. Non-map → identity.
        By::Column(col, _) => match value {
            GVal::Map(entries) => GVal::List(
                entries
                    .iter()
                    .map(|(k, v)| {
                        if *col == Column::Keys {
                            k.clone()
                        } else {
                            v.clone()
                        }
                    })
                    .collect(),
            ),
            _ => value.clone(),
        },
        By::Traversal(plan, _) => sub_vals(graph, ctx, plan, &Trav::root(value.clone()))
            .into_iter()
            .next()
            .unwrap_or(GVal::Null),
    }
}

/// Does this sub-plan end in a reducing barrier? Such a value-by folds a whole
/// group to a single value (`count`/`sum`/`min`/`max`/`mean`/`fold`); any other
/// maps each member and its outputs are collected into a list.
fn is_reducing(steps: &[Step]) -> bool {
    matches!(
        steps.last(),
        Some(
            Step::Count(_)
                | Step::Sum(_)
                | Step::Min(_)
                | Step::Max(_)
                | Step::Mean(_)
                | Step::Fold
        )
    )
}

/// The value a `group()` bucket maps to. A traversal value-by is applied over the
/// group's MEMBERS as a barrier (so `count()` counts the group, not each element):
/// a reducing traversal yields the single folded value; a mapping one collects
/// every output. A non-traversal by (identity/key/token) maps each member.
fn group_value(graph: &mut Graph, ctx: &mut Ctx, by: &By, members: Vec<Trav>) -> GVal {
    match by {
        By::Traversal(plan, _) => {
            let outs: Vec<GVal> = run_steps(graph, ctx, &plan.steps, members)
                .into_iter()
                .map(|t| t.val)
                .collect();
            if is_reducing(&plan.steps) {
                outs.into_iter().next().unwrap_or(GVal::Null)
            } else {
                GVal::list(outs)
            }
        }
        _ => GVal::List(
            members
                .iter()
                .map(|t| eval_by(graph, ctx, by, &t.val))
                .collect(),
        ),
    }
}

/// Elements of a value for `Scope::local` (non-string iterables; else singleton).
fn local_elems(v: &GVal) -> Vec<GVal> {
    match v {
        GVal::List(items) => items.to_vec(),
        other => vec![other.clone()],
    }
}

// --- per-step application ---------------------------------------------------

fn apply(graph: &mut Graph, ctx: &mut Ctx, step: &Step, stream: Vec<Trav>) -> Vec<Trav> {
    match step {
        // --- sources (root) / re-source (mid-traversal, carrying tags) ---
        Step::V(ids) => {
            let verts: Vec<u32> = if ids.is_empty() {
                graph.vertex_indices().collect()
            } else {
                ids.iter().filter_map(|id| graph.vid.get(id)).filter(|&v| graph.is_vertex_live(v)).collect()
            };
            if stream.is_empty() {
                verts.into_iter().map(|v| Trav::root(GVal::Node(v))).collect()
            } else {
                stream.iter().flat_map(|t| verts.iter().map(move |&v| t.step(GVal::Node(v)))).collect()
            }
        }
        Step::E(ids) => {
            // Match an external edge id (e.g. `E('7')`, like `V('1')`), falling
            // back to the synthetic `e{index}` form when no external id was set.
            let edges: Vec<u32> = if ids.is_empty() {
                (0..graph.e_src.len() as u32).filter(|&e| graph.is_edge_live(e)).collect()
            } else {
                // Resolve each id DIRECTLY (like `V(ids)`) — O(ids), not the old
                // O(edges × ids) scan of every edge. `edge_by_id` resolves both an
                // assigned id and the canonical `e{index}` form. This also matches the
                // TS engine, which yields per requested id in id order (no dedup) — the
                // old full scan emitted in edge-index order and deduped.
                ids.iter()
                    .filter_map(|i| graph.edge_by_id(i))
                    .filter(|&e| graph.is_edge_live(e))
                    .collect()
            };
            if stream.is_empty() {
                edges.into_iter().map(|e| Trav::root(GVal::Edge(e))).collect()
            } else {
                stream.iter().flat_map(|t| edges.iter().map(move |&e| t.step(GVal::Edge(e)))).collect()
            }
        }

        // --- vertex → vertex ---
        Step::Out(labels) | Step::In(labels) | Step::Both(labels) => {
            let (out, inn) = dir_flags(step);
            let dir = seek_dir(out, inn);
            let Some(etypes) = resolve_etypes(graph, labels) else {
                return Vec::new(); // a label naming no type matches nothing
            };
            let mut next = Vec::new();

            for t in &stream {
                if let GVal::Node(v) = t.val {
                    next.extend(
                        crate::seek::adj(graph, v, dir, &etypes, crate::seek::SelfLoops::Twice)
                            .map(|a| t.step(GVal::Node(a.nbr))),
                    );
                }
            }

            next
        }

        // --- vertex → edge ---
        Step::OutE(labels) | Step::InE(labels) | Step::BothE(labels) => {
            let (out, inn) = dir_flags(step);
            let dir = seek_dir(out, inn);
            let Some(etypes) = resolve_etypes(graph, labels) else {
                return Vec::new();
            };
            let mut next = Vec::new();

            for t in &stream {
                if let GVal::Node(v) = t.val {
                    next.extend(
                        crate::seek::adj(graph, v, dir, &etypes, crate::seek::SelfLoops::Twice)
                            .map(|a| t.step(GVal::Edge(a.eidx))),
                    );
                }
            }

            next
        }

        // --- edge → vertex ---
        Step::OutV => map_step(stream, |t| match t.val {
            GVal::Edge(e) => vec![GVal::Node(graph.e_src[e as usize])],
            _ => vec![],
        }),
        Step::InV => map_step(stream, |t| match t.val {
            GVal::Edge(e) => vec![GVal::Node(graph.e_dst[e as usize])],
            _ => vec![],
        }),
        Step::BothV => map_step(stream, |t| match t.val {
            GVal::Edge(e) => vec![GVal::Node(graph.e_src[e as usize]), GVal::Node(graph.e_dst[e as usize])],
            _ => vec![],
        }),
        Step::OtherV => {
            let mut next = Vec::new();
            for t in &stream {
                if let GVal::Edge(e) = t.val {
                    let (src, dst) = (graph.e_src[e as usize], graph.e_dst[e as usize]);
                    let from = t.path.iter().rev().nth(1).and_then(|g| match g {
                        GVal::Node(v) => Some(*v),
                        _ => None,
                    });
                    next.push(t.step(GVal::Node(if from == Some(src) { dst } else { src })));
                }
            }
            next
        }

        // --- filters ---
        Step::Has(key, pred) => stream.into_iter().filter(|t| p_matches(pred, &prop(graph, &t.val, key))).collect(),
        // Resolve the wanted labels to IDS once, then compare ids per element.
        // This used to call `elem_label`, which builds an `Arc<str>` per element
        // (a refcount bump plus a `GVal`) and string-compares it — on a 50k-vertex
        // scan that is 50k allocations to answer a question about two integers.
        //
        // Semantics are unchanged and deliberately narrow: `hasLabel` here matches
        // an element's FIRST label only (`vertex_labels(i).first()`), not any of
        // them, so this compares that same one. It is also why the label BUCKET
        // (`by_label`, which indexes a vertex under every label it carries) is NOT
        // a valid shortcut — it would match a `[A, B]` vertex on `hasLabel('B')`
        // where the traversal does not.
        Step::HasLabel(labels) => {
            let want: Vec<u32> = labels.iter().filter_map(|l| graph.labels.get(l)).collect();
            let want_e: Vec<u32> = labels.iter().filter_map(|l| graph.etype.get(l)).collect();

            stream
                .into_iter()
                .filter(|t| match &t.val {
                    GVal::Node(i) => graph
                        .vertex_labels(*i)
                        .first()
                        .is_some_and(|lid| want.contains(lid)),
                    GVal::Edge(e) => want_e.contains(&graph.e_type[*e as usize]),
                    _ => false,
                })
                .collect()
        }
        Step::HasId(ids) => stream.into_iter().filter(|t| matches!(elem_id(graph, &t.val), GVal::Str(ref s) if ids.iter().any(|i| i == s.as_ref()))).collect(),
        Step::HasKey(keys) => stream
            .into_iter()
            .filter(|t| {
                let present = present_keys(graph, &t.val);
                keys.iter().any(|k| present.iter().any(|p| p == k))
            })
            .collect(),
        Step::HasNot(keys) => stream
            .into_iter()
            .filter(|t| {
                let present = present_keys(graph, &t.val);
                !keys.iter().any(|k| present.iter().any(|p| p == k))
            })
            .collect(),
        Step::HasValue(vals) => stream.into_iter().filter(|t| prop_value_field(&t.val).is_some_and(|v| vals.contains(&v))).collect(),
        Step::Is(pred) => stream.into_iter().filter(|t| p_matches(pred, &t.val)).collect(),
        Step::SimplePath => stream.into_iter().filter(|t| !has_dup(&t.path)).collect(),
        Step::CyclicPath => stream.into_iter().filter(|t| has_dup(&t.path)).collect(),
        Step::Dedupe { labels, bys } => apply_dedupe(graph, ctx, labels, bys, stream),

        // --- projection ---
        Step::Values(keys) => {
            let mut next = Vec::new();

            // One key is the common shape, and `value`/`is_present` hash the key
            // name on EVERY call — 400k dictionary lookups to read one property
            // off a 200k stream. `value_id`/`is_present_id` are the same reads
            // against an already-resolved column, so resolve it once here.
            if let [only] = keys.as_slice() {
                let vk = graph.props.keys.get(only);
                let ek = graph.edge_props.keys.get(only);

                for t in &stream {
                    match (&t.val, vk, ek) {
                        (GVal::Node(i), Some(k), _) => {
                            let i = *i as usize;

                            if graph.props.is_present_id(i, k) {
                                let v = graph.props.value_id(i, k, &graph.strs);

                                next.push(t.step(value_to_gval(v)));
                            }
                        }
                        (GVal::Edge(e), _, Some(k)) => {
                            let e = *e as usize;

                            if graph.edge_props.is_present_id(e, k) {
                                let v = graph.edge_props.value_id(e, k, &graph.strs);

                                next.push(t.step(value_to_gval(v)));
                            }
                        }
                        // A key the store never saw, or a non-element value:
                        // nothing to read.
                        _ => {}
                    }
                }

                return next;
            }

            for t in &stream {
                // `values()` with no argument needs a per-element key list,
                // because the keys differ per element. It used to CLONE the
                // argument list per traverser even when there was one.
                if keys.is_empty() {
                    for k in present_keys(graph, &t.val) {
                        // Gate on PRESENCE, not value != Null: a present null
                        // yields a `Null` here; only an absent key is skipped.
                        if prop_present(graph, &t.val, &k) {
                            next.push(t.step(prop(graph, &t.val, &k)));
                        }
                    }
                } else {
                    for k in keys {
                        if prop_present(graph, &t.val, k) {
                            next.push(t.step(prop(graph, &t.val, k)));
                        }
                    }
                }
            }

            next
        }
        Step::ValueMap(keys) => {
            // Same shape as `values`: the key list was CLONED per traverser and
            // every read hashed the key NAME, so a 200k stream paid 200k
            // `Vec<String>` clones plus two dictionary lookups per property.
            // Resolving the names — and the `Arc<str>` each becomes — once here
            // leaves an array index per read.
            let named: Vec<(Arc<str>, Option<u32>, Option<u32>)> = keys
                .iter()
                .map(|k| {
                    (
                        Arc::from(k.as_str()),
                        graph.props.keys.get(k),
                        graph.edge_props.keys.get(k),
                    )
                })
                .collect();

            map_step(stream, |t| {
                let entries: Vec<(GVal, GVal)> = if keys.is_empty() {
                    present_key_ids(graph, &t.val)
                        .into_iter()
                        .map(|(kid, k)| (GVal::Str(k), prop_by_id(graph, &t.val, kid)))
                        .collect()
                } else {
                    named
                        .iter()
                        .filter_map(|(name, vk, ek)| {
                            let (store, kid) = match (&t.val, vk, ek) {
                                (GVal::Node(i), Some(k), _) => {
                                    (&graph.props, (*i as usize, *k))
                                }
                                (GVal::Edge(e), _, Some(k)) => {
                                    (&graph.edge_props, (*e as usize, *k))
                                }
                                _ => return None,
                            };

                            // Presence, not nullness: a stored null rides through.
                            store.is_present_id(kid.0, kid.1).then(|| {
                                (
                                    GVal::Str(name.clone()),
                                    value_to_gval(store.value_id(kid.0, kid.1, &graph.strs)),
                                )
                            })
                        })
                        .collect()
                };

                vec![GVal::Map(entries)]
            })
        }
        Step::PropertyMap(keys) => map_step(stream, |t| {
            let ks = if keys.is_empty() { present_keys(graph, &t.val) } else { keys.clone() };
            let entries = ks
                .into_iter()
                .filter(|k| prop_present(graph, &t.val, k))
                .map(|k| (GVal::Str(Arc::from(k.as_str())), GVal::list(vec![prop(graph, &t.val, &k)])))
                .collect();
            vec![GVal::Map(entries)]
        }),
        Step::ElementMap(keys) => map_step(stream, |t| {
            if !matches!(t.val, GVal::Node(_) | GVal::Edge(_)) {
                return vec![];
            }
            let mut entries = vec![
                (GVal::Str(Arc::from("id")), elem_id(graph, &t.val)),
                (GVal::Str(Arc::from("label")), elem_label(graph, &t.val)),
            ];
            if let GVal::Edge(e) = t.val {
                let inv = GVal::Node(graph.e_dst[e as usize]);
                let outv = GVal::Node(graph.e_src[e as usize]);
                entries.push((GVal::Str(Arc::from("IN")), GVal::Map(vec![(GVal::Str(Arc::from("id")), elem_id(graph, &inv)), (GVal::Str(Arc::from("label")), elem_label(graph, &inv))])));
                entries.push((GVal::Str(Arc::from("OUT")), GVal::Map(vec![(GVal::Str(Arc::from("id")), elem_id(graph, &outv)), (GVal::Str(Arc::from("label")), elem_label(graph, &outv))])));
            }
            let ks = if keys.is_empty() { present_keys(graph, &t.val) } else { keys.clone() };
            for k in ks {
                if prop_present(graph, &t.val, &k) {
                    entries.push((GVal::Str(Arc::from(k.as_str())), prop(graph, &t.val, &k)));
                }
            }
            vec![GVal::Map(entries)]
        }),
        Step::Properties(keys) => {
            let mut next = Vec::new();
            for t in &stream {
                let ks = if keys.is_empty() { present_keys(graph, &t.val) } else { keys.clone() };
                for k in ks {
                    if prop_present(graph, &t.val, &k) {
                        let v = prop(graph, &t.val, &k);
                        next.push(t.step(GVal::property(
                            t.val.clone(),
                            Arc::from(k.as_str()),
                            v,
                        )));
                    }
                }
            }
            next
        }
        // A property element yields its value; any other value passes through
        // unchanged (identity), matching the TS engine.
        Step::Value => map_step(stream, |t| vec![prop_value_field(&t.val).unwrap_or_else(|| t.val.clone())]),
        Step::Id => map_step(stream, |t| vec![elem_id(graph, &t.val)]),
        Step::Label => map_step(stream, |t| match &t.val {
            // A property element's `label()` is its key (TinkerPop).
            GVal::Property { .. } | GVal::Map(_) => {
                prop_key_field(&t.val).map(|v| vec![v]).unwrap_or_default()
            }
            other => vec![elem_label(graph, other)],
        }),
        Step::Path(bys) => stream
            .iter()
            .map(|t| {
                let projected = if bys.is_empty() {
                    t.path.clone()
                } else {
                    t.path.iter().enumerate().map(|(i, v)| eval_by(graph, ctx, &bys[i % bys.len()], v)).collect()
                };
                t.with(GVal::list(projected))
            })
            .collect(),
        Step::Project(keys, bys) => stream
            .iter()
            .map(|t| {
                let entries = keys
                    .iter()
                    .enumerate()
                    .map(|(i, k)| {
                        let v = match bys.get(i) {
                            Some(by) => eval_by(graph, ctx, by, &t.val),
                            None => t.val.clone(),
                        };
                        (GVal::Str(Arc::from(k.as_str())), v)
                    })
                    .collect();
                t.with(GVal::Map(entries))
            })
            .collect(),
        Step::Tree(bys) => apply_tree(graph, ctx, bys, stream),

        // --- cardinality ---
        Step::Limit(n, Scope::Global) => stream.into_iter().take(*n).collect(),
        Step::Limit(n, Scope::Local) => map_step(stream, |t| vec![slice_local(&t.val, 0, *n)]),
        Step::Skip(n, Scope::Global) => stream.into_iter().skip(*n).collect(),
        Step::Skip(n, Scope::Local) => map_step(stream, |t| vec![slice_local(&t.val, *n, usize::MAX)]),
        Step::Range(s, e, Scope::Global) => stream.into_iter().skip(*s).take(e.saturating_sub(*s)).collect(),
        Step::Range(s, e, Scope::Local) => map_step(stream, |t| vec![slice_local(&t.val, *s, *e)]),
        Step::Tail(n, Scope::Global) => {
            let len = stream.len();
            stream.into_iter().skip(len.saturating_sub(*n)).collect()
        }
        Step::Tail(n, Scope::Local) => map_step(stream, |t| {
            let e = local_elems(&t.val);
            let start = e.len().saturating_sub(*n);
            vec![GVal::list(e[start..].to_vec())]
        }),
        Step::Sample(n) => apply_sample(*n, stream),

        // --- aggregates ---
        Step::Count(Scope::Global) => vec![Trav::root(GVal::Num(stream.len() as f64))],
        Step::Count(Scope::Local) => map_step(stream, |t| vec![GVal::Num(local_elems(&t.val).len() as f64)]),
        Step::Fold => vec![Trav::root(GVal::List(stream.into_iter().map(|t| t.val).collect()))],
        Step::Sum(Scope::Global) => fold_num(stream, |ns| ns.iter().sum()),
        Step::Sum(Scope::Local) => map_step(stream, |t| vec![local_num(&t.val, |ns| ns.iter().sum())]),
        Step::Mean(Scope::Global) => fold_num(stream, |ns| ns.iter().sum::<f64>() / ns.len() as f64),
        Step::Mean(Scope::Local) => map_step(stream, |t| vec![local_num(&t.val, |ns| ns.iter().sum::<f64>() / ns.len() as f64)]),
        Step::Min(Scope::Global) => fold_extreme(stream, Ordering::Less),
        Step::Min(Scope::Local) => map_step(stream, |t| vec![local_extreme(&t.val, Ordering::Less)]),
        Step::Max(Scope::Global) => fold_extreme(stream, Ordering::Greater),
        Step::Max(Scope::Local) => map_step(stream, |t| vec![local_extreme(&t.val, Ordering::Greater)]),
        Step::Order(bys, desc, scope) => apply_order(graph, ctx, bys, *desc, scope, stream),
        Step::Group(bys) => {
            let key_by = bys.first().cloned().unwrap_or(By::Identity(None));
            let val_by = bys.get(1).cloned().unwrap_or(By::Identity(None));
            // Bucket the group's MEMBERS (traversers), keeping key + insertion
            // order, so a reducing value-by can fold over each group as a barrier.
            let mut buckets: Vec<(GVal, Vec<Trav>)> = Vec::new();
            // O(1) key->bucket index beside the insertion-ordered Vec (output order
            // is the Vec's, unchanged). A NaN-bearing key has no DedupKey, so it
            // falls back to the linear scan — matching the old exact `GVal` equality.
            let mut index: HashMap<DedupKey, usize> = HashMap::new();
            for t in stream {
                let key = eval_by(graph, ctx, &key_by, &t.val);
                let dk = dedup_key(&key);
                let existing = match &dk {
                    Some(dk) => index.get(dk).copied(),
                    None => buckets.iter().position(|(k, _)| *k == key),
                };
                match existing {
                    Some(i) => buckets[i].1.push(t),
                    None => {
                        if let Some(dk) = dk {
                            index.insert(dk, buckets.len());
                        }
                        buckets.push((key, vec![t]));
                    }
                }
            }
            let entries: Vec<(GVal, GVal)> = buckets
                .into_iter()
                .map(|(k, members)| (k, group_value(graph, ctx, &val_by, members)))
                .collect();
            vec![Trav::root(GVal::Map(entries))]
        }
        Step::GroupCount(bys) => {
            let by = bys.first().cloned().unwrap_or(By::Identity(None));
            let mut entries: Vec<(GVal, f64)> = Vec::new();
            // O(1) key->slot index beside the insertion-ordered Vec (see Group above).
            let mut index: HashMap<DedupKey, usize> = HashMap::new();
            for t in &stream {
                let key = eval_by(graph, ctx, &by, &t.val);
                let dk = dedup_key(&key);
                let existing = match &dk {
                    Some(dk) => index.get(dk).copied(),
                    None => entries.iter().position(|(k, _)| *k == key),
                };
                match existing {
                    Some(i) => entries[i].1 += 1.0,
                    None => {
                        if let Some(dk) = dk {
                            index.insert(dk, entries.len());
                        }
                        entries.push((key, 1.0));
                    }
                }
            }
            vec![Trav::root(GVal::Map(entries.into_iter().map(|(k, n)| (k, GVal::Num(n))).collect()))]
        }

        // --- combinators ---
        Step::Where(sub) => stream.into_iter().filter(|t| sub_nonempty(graph, ctx, sub, t)).collect(),
        Step::WhereKey(start, pred, bys) => {
            let Some(GVal::Str(end_label)) = pred.rhs() else {
                return stream; // non-comparison predicate; nothing to compare against
            };
            let end_label = end_label.to_string();
            let start_by = bys.first().cloned().unwrap_or(By::Identity(None));
            let end_by = bys.get(1).cloned().unwrap_or_else(|| start_by.clone());
            let mut next = Vec::new();
            for t in stream {
                let (Some(sv), Some(ev)) = (t.recall(start, Pop::Last), t.recall(&end_label, Pop::Last)) else {
                    continue;
                };
                let sv = eval_by(graph, ctx, &start_by, &sv);
                let ev = eval_by(graph, ctx, &end_by, &ev);
                let resolved = substitute_rhs(pred, ev);
                if p_matches(&resolved, &sv) {
                    next.push(t);
                }
            }
            next
        }
        Step::WherePred(pred) => {
            // Predicate-only `where(neq('me'))`: compare the CURRENT traverser value
            // against the value tagged at the predicate's step-label operand.
            let Some(GVal::Str(label)) = pred.rhs() else {
                return stream; // non-comparison predicate; nothing to compare against
            };
            let label = label.to_string();
            let mut next = Vec::new();
            for t in stream {
                let Some(ev) = t.recall(&label, Pop::Last) else {
                    continue; // the referenced label isn't bound → drop, as WhereKey does
                };
                let resolved = substitute_rhs(pred, ev);
                if p_matches(&resolved, &t.val) {
                    next.push(t);
                }
            }
            next
        }
        Step::And(plans) => stream.into_iter().filter(|t| plans.iter().all(|p| sub_nonempty(graph, ctx, p, t))).collect(),
        Step::Or(plans) => stream.into_iter().filter(|t| plans.iter().any(|p| sub_nonempty(graph, ctx, p, t))).collect(),
        Step::Not(sub) => stream.into_iter().filter(|t| !sub_nonempty(graph, ctx, sub, t)).collect(),
        Step::Union(plans) => {
            let mut next = Vec::new();
            for t in &stream {
                for p in plans {
                    next.extend(run_steps(graph, ctx, &p.steps, vec![t.clone()]));
                }
            }
            next
        }
        Step::Coalesce(plans) => {
            let mut next = Vec::new();
            for t in &stream {
                for p in plans {
                    let r = run_steps(graph, ctx, &p.steps, vec![t.clone()]);
                    if !r.is_empty() {
                        next.extend(r);
                        break;
                    }
                }
            }
            next
        }
        Step::Optional(sub) => {
            let mut next = Vec::new();
            for t in stream {
                let r = run_steps(graph, ctx, &sub.steps, vec![t.clone()]);
                if r.is_empty() {
                    next.push(t);
                } else {
                    next.extend(r);
                }
            }
            next
        }
        Step::Local(sub) => {
            let mut next = Vec::new();
            for t in &stream {
                next.extend(run_steps(graph, ctx, &sub.steps, vec![t.clone()]));
            }
            next
        }
        Step::Choose { test, then_, else_ } => {
            let mut next = Vec::new();
            for t in stream {
                if sub_nonempty(graph, ctx, test, &t) {
                    next.extend(run_steps(graph, ctx, &then_.steps, vec![t]));
                } else if let Some(e) = else_ {
                    next.extend(run_steps(graph, ctx, &e.steps, vec![t]));
                } else {
                    next.push(t);
                }
            }
            next
        }
        Step::Branch {
            test,
            options,
            default,
        } => {
            let mut next = Vec::new();
            for t in stream {
                // Route by the test plan's first result: the first option whose
                // `match` equals it, else the default (if any).
                let tv = sub_vals(graph, ctx, test, &t).into_iter().next();
                let mut target: Option<&Traversal> = None;
                if let Some(ref v) = tv {
                    for (m, plan) in options {
                        if m == v {
                            target = Some(plan);
                            break;
                        }
                    }
                }
                if let Some(plan) = target.or(default.as_deref()) {
                    next.extend(run_steps(graph, ctx, &plan.steps, vec![t]));
                }
            }
            next
        }
        Step::Map(sub) => {
            let mut next = Vec::new();
            for t in &stream {
                if let Some(v) = sub_vals(graph, ctx, sub, t).into_iter().next() {
                    next.push(t.with(v));
                }
            }
            next
        }
        Step::FlatMap(sub) => {
            let mut next = Vec::new();
            for t in &stream {
                for v in sub_vals(graph, ctx, sub, t) {
                    next.push(t.with(v));
                }
            }
            next
        }
        Step::SideEffect(sub) => {
            for t in &stream {
                let _ = run_steps(graph, ctx, &sub.steps, vec![t.clone()]);
            }
            stream
        }
        Step::Aggregate(key) | Step::Store(key) => {
            for t in &stream {
                ctx.side.entry(key.clone()).or_default().push(t.val.clone());
            }
            stream
        }
        Step::Subgraph(key) => {
            // Accumulate each edge (+ its endpoints) into the named subgraph,
            // deduped by id; traversers pass through so it composes mid-stream.
            let entry = ctx.subgraphs.entry(key.clone()).or_default();
            for t in &stream {
                if let GVal::Edge(e) = t.val {
                    let (s, d) = (graph.e_src[e as usize], graph.e_dst[e as usize]);
                    if !entry.1.contains(&e) {
                        entry.1.push(e);
                    }
                    for v in [s, d] {
                        if !entry.0.contains(&v) {
                            entry.0.push(v);
                        }
                    }
                }
            }
            stream
        }
        Step::ShortestPath { target, out, inn } => {
            shortest_path_step(graph, ctx, target.as_deref(), *out, *inn, stream)
        }
        Step::PageRank {
            property,
            times,
            alpha,
        } => algo_step(
            graph,
            ctx,
            stream,
            "pagerank",
            property
                .clone()
                .unwrap_or_else(|| "gremlin.pageRankVertexProgram.pageRank".to_string()),
            |cfg| {
                cfg.iterations = *times;
                cfg.damping_factor = *alpha;
            },
        ),
        Step::ConnectedComponent { property } => algo_step(
            graph,
            ctx,
            stream,
            "connectedComponents",
            property
                .clone()
                .unwrap_or_else(|| "gremlin.connectedComponentVertexProgram.component".to_string()),
            |_| {},
        ),
        Step::PeerPressure { property, times } => algo_step(
            graph,
            ctx,
            stream,
            "peerPressure",
            property
                .clone()
                .unwrap_or_else(|| "gremlin.peerPressureVertexProgram.cluster".to_string()),
            |cfg| cfg.iterations = *times,
        ),
        Step::Cap(key) => {
            // A subgraph key caps to a self-describing {vertices, edges} map of
            // full element records (GVal has no graph type — the TS engine returns
            // a Graph object). The JS `subgraphToGraph` helper rebuilds a real
            // @lenke/core Graph from this, giving cross-engine parity. Else the
            // capped value is the plain side-effect bag.
            if let Some((verts, edges)) = ctx.subgraphs.get(key) {
                let (verts, edges) = (verts.clone(), edges.clone());
                let vlist = GVal::List(verts.iter().map(|v| subgraph_vertex(graph, *v)).collect());
                let elist = GVal::List(edges.iter().map(|e| subgraph_edge(graph, *e)).collect());
                vec![Trav::root(GVal::Map(vec![
                    (GVal::Str(Arc::from("vertices")), vlist),
                    (GVal::Str(Arc::from("edges")), elist),
                ]))]
            } else {
                vec![Trav::root(GVal::list(ctx.side.get(key).cloned().unwrap_or_default()))]
            }
        }
        Step::Barrier => stream,
        Step::Repeat { body, times, until, until_before, emit, emit_before } => run_repeat(
            graph,
            ctx,
            &stream,
            body,
            RepeatMods {
                times: *times,
                until: until.as_deref(),
                until_before: *until_before,
                emit: emit.as_deref(),
                emit_before: *emit_before,
            },
        ),

        // --- tagging / select ---
        Step::As(label) => stream
            .into_iter()
            .map(|mut t| {
                let val = t.val.clone();
                let tags = Arc::make_mut(&mut t.tags);
                match tags.iter_mut().find(|(l, _)| l == label) {
                    Some((_, list)) => list.push(val),
                    None => tags.push((label.clone(), vec![val])),
                }
                t
            })
            .collect(),
        Step::SelectColumn(col) => {
            // Extract a Map's keys or values as a list, preserving entry order (so a
            // preceding `order(local)` is observable). A non-map traverser is dropped,
            // as TinkerPop's `select(Column)` filters it.
            let mut next = Vec::new();
            for t in stream {
                if let GVal::Map(entries) = &t.val {
                    let list = entries
                        .iter()
                        .map(|(k, v)| if *col == Column::Keys { k.clone() } else { v.clone() })
                        .collect();
                    next.push(t.step(GVal::List(list)));
                }
            }
            next
        }
        Step::Select { labels, pop, bys } => {
            let mut next = Vec::new();
            for t in &stream {
                // A label resolves against the path tags first; failing that, if the
                // current value is a Map (a `project()`/`valueMap()` row), `select(k)`
                // projects the entry at that key — TinkerPop's `Scoping` semantics, and
                // the reason `project(...).order().by(select('k'))` sorts rather than
                // silently no-op'ing (the sub-`select` ran on a fresh, untagged root
                // traverser, so a tag-only lookup dropped every row). Both engines apply
                // the same fallback, so results stay byte-identical.
                let vals: Vec<Option<GVal>> = labels
                    .iter()
                    .map(|l| {
                        t.recall(l, *pop).or_else(|| match &t.val {
                            GVal::Map(entries) => entries
                                .iter()
                                .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == l.as_str()))
                                .map(|(_, v)| v.clone()),
                            _ => None,
                        })
                    })
                    .collect();
                if vals.iter().any(Option::is_none) {
                    continue;
                }
                // A single `by()` cycles across all labels (Gremlin semantics); no
                // `by()` ⇒ identity. Matches the TS selectStep.
                let by_at = |i: usize| -> By {
                    if bys.is_empty() {
                        By::Identity(None)
                    } else {
                        bys[i % bys.len()].clone()
                    }
                };
                if labels.len() == 1 {
                    let v = eval_by(graph, ctx, &by_at(0), vals[0].as_ref().unwrap());
                    next.push(t.with(v));
                } else {
                    let entries = labels
                        .iter()
                        .enumerate()
                        .map(|(i, l)| {
                            (GVal::Str(Arc::from(l.as_str())), eval_by(graph, ctx, &by_at(i), vals[i].as_ref().unwrap()))
                        })
                        .collect();
                    next.push(t.with(GVal::Map(entries)));
                }
            }
            next
        }
        Step::Match(plans) => match_step(graph, ctx, plans, stream),

        // --- misc ---
        Step::Unfold => {
            let mut next = Vec::new();
            for t in &stream {
                match &t.val {
                    GVal::List(items) => {
                        for it in items.iter() {
                            next.push(t.step(it.clone()));
                        }
                    }
                    other => next.push(t.step(other.clone())),
                }
            }
            next
        }
        Step::Index => stream.iter().enumerate().map(|(i, t)| t.with(GVal::list(vec![t.val.clone(), GVal::Num(i as f64)]))).collect(),
        Step::Loops => map_step(stream, |t| vec![GVal::Num(t.loops as f64)]),
        Step::Constant(v) => map_step(stream, |_t| vec![v.clone()]),
        Step::Math { expr, bys } => {
            let vars = math_vars(expr);
            let mut next = Vec::with_capacity(stream.len());
            for t in &stream {
                // Resolve each operand to a number: `_` is the current value, any
                // other name is an `as_`-bound tag; project via the cycling by()s.
                let mut vals: HashMap<String, f64> = HashMap::new();
                let mut ok = true;
                for (i, name) in vars.iter().enumerate() {
                    let base = if name == "_" {
                        Some(t.val.clone())
                    } else {
                        t.recall(name, Pop::Last)
                    };
                    let Some(base) = base else {
                        // Unbound: fine if it's a constant (`pi`/`e`) or a bare
                        // function name (`sin _`) — the parser supplies the value
                        // and neither takes a by() projection.
                        if math_const(name).is_some() || unary_math_fn(name).is_some() {
                            continue;
                        }
                        set_type_fault(); // unbound variable
                        ok = false;
                        break;
                    };
                    let by = if bys.is_empty() {
                        By::Identity(None)
                    } else {
                        bys[i % bys.len()].clone()
                    };
                    let projected = eval_by(graph, ctx, &by, &base);
                    match strict_num(&projected) {
                        Some(n) => {
                            vals.insert(name.clone(), n);
                        }
                        None => {
                            set_type_fault(); // null / non-numeric operand
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }
                match eval_math_expr(expr, &vals) {
                    Some(r) => next.push(t.with(GVal::Num(r))),
                    None => set_type_fault(), // malformed expression
                }
            }
            next
        }
        Step::Identity => stream,
        Step::Inject(vs) => {
            let mut next: Vec<Trav> = vs.iter().map(|v| Trav::root(v.clone())).collect();
            next.extend(stream);
            next
        }
        Step::None(None) => Vec::new(),
        Step::None(Some(pred)) => stream.into_iter().filter(|t| !local_elems(&t.val).iter().any(|e| p_matches(pred, e))).collect(),
        Step::Fail(msg) => {
            // `fail()` on a non-empty stream is a user-raised runtime exception:
            // record it as a data fault (surfaced by `try_run`, ignored by `run`)
            // and drop the stream — never `panic!`, which under the release
            // `panic = "abort"` build with no FFI `catch_unwind` would abort the
            // whole host process. The TS engine throws a catchable error here.
            if stream.is_empty() {
                stream
            } else {
                ctx.fault.get_or_insert((
                    crate::error_codes::ErrorCode::DataException,
                    msg.clone().map_or(std::borrow::Cow::Borrowed("fail() reached"), std::borrow::Cow::Owned),
                ));
                Vec::new()
            }
        }

        // --- mutation ---
        Step::AddV(label) => {
            let labels: Vec<String> = label.iter().cloned().collect();
            // A malformed label (empty / contains `::`) is a data fault — Gremlin
            // takes arbitrary label strings, so guard here (codec ingestion has
            // its own gate). `run` returns nothing; `try_run` surfaces the fault.
            if labels.iter().any(|l| crate::graph::validate_label(l).is_err()) {
                ctx.fault.get_or_insert((
                    crate::error_codes::ErrorCode::InvalidValue,
                    "addV(): a label must be non-empty and cannot contain '::'".into(),
                ));
                return Vec::new();
            }
            // As a source (`g.addV()`), create one even with no incoming traverser.
            let base = if stream.is_empty() { vec![Trav::root(GVal::Null)] } else { stream };
            base.iter().map(|t| t.with(GVal::Node(graph.add_vertex(&labels, vec![])))).collect()
        }
        Step::AddE { label, from, to } => {
            if crate::graph::validate_label(label).is_err() {
                ctx.fault.get_or_insert((
                    crate::error_codes::ErrorCode::InvalidValue,
                    "addE(): a label must be non-empty and cannot contain '::'".into(),
                ));
                return Vec::new();
            }
            let mut next = Vec::new();
            for t in &stream {
                let (Some(f), Some(to_v)) = (resolve_endpoint(graph, ctx, from, t), resolve_endpoint(graph, ctx, to, t)) else {
                    // An unresolvable endpoint is a data fault (TS throws
                    // MissingVertex), not a silent drop.
                    ctx.fault.get_or_insert((
                        crate::error_codes::ErrorCode::MissingVertex,
                        "addE(): could not resolve endpoint vertices".into(),
                    ));
                    continue;
                };
                let e = graph.add_edge(f, to_v, label, vec![]);
                next.push(t.with(GVal::Edge(e)));
            }
            next
        }
        Step::Property(key, _) if crate::graph::validate_prop_key(key).is_err() => {
            ctx.fault.get_or_insert((
                crate::error_codes::ErrorCode::InvalidValue,
                "property(): a key must be non-empty".into(),
            ));
            Vec::new()
        }
        Step::Property(key, v) => {
            let mut next = Vec::with_capacity(stream.len());
            for t in stream {
                // A traversal value is re-evaluated per element, rooted at the
                // current traverser (so it can `select(...)` an outer label); its
                // first output is the value, no output leaves the property unset.
                let val = match v {
                    PropVal::Lit(g) => Some(g.clone()),
                    PropVal::Trav(plan) => sub_vals(graph, ctx, plan, &t).into_iter().next(),
                };
                let target = match &t.val {
                    GVal::Node(i) => Some((true, *i)),
                    GVal::Edge(e) => Some((false, *e)),
                    _ => None,
                };
                match (target, val) {
                    (Some((true, i)), Some(g)) => {
                        graph.set_vertex_prop(i, key, gval_to_value(&g));
                        next.push(t);
                    }
                    (Some((false, e)), Some(g)) => {
                        graph.set_edge_prop(e, key, gval_to_value(&g));
                        next.push(t);
                    }
                    // property() on a non-element (or a traversal that produced no
                    // value) drops the traverser (matches TS), not pass-through.
                    _ => {}
                }
            }
            next
        }
        Step::Drop => {
            for t in &stream {
                match &t.val {
                    GVal::Node(i) => {
                        let _ = graph.remove_vertex(*i, true);
                    }
                    GVal::Edge(e) => {
                        graph.remove_edge(*e);
                    }
                    // `.properties(k).drop()` removes the property from its owner.
                    // This is the ONLY way to delete a property in Gremlin here —
                    // `property(k, null)` STORES a null (divergence from TinkerPop).
                    // The owner is carried on the element itself, so a `project`
                    // Map (not a Property) can never be mistaken for one.
                    GVal::Property(p) => match p.owner {
                        GVal::Node(i) => graph.remove_vertex_prop(i, &p.key),
                        GVal::Edge(e) => graph.remove_edge_prop(e, &p.key),
                        _ => {}
                    },
                    _ => {}
                }
            }
            Vec::new()
        }
        // `withSack(init)` just records the default; no sack is created until a
        // read/write actually touches one (laziness).
        Step::WithSack(init) => {
            ctx.sack_init = Some(init.clone());
            stream
        }
        Step::Sack { op, bys } => {
            let Some(default) = ctx.sack_init.clone() else {
                ctx.fault.get_or_insert((
                    crate::error_codes::ErrorCode::InvalidGraphOp,
                    "sack() requires a preceding withSack()".into(),
                ));
                return Vec::new();
            };
            match op {
                // `sack()` — emit the current sack as the traverser's value (the
                // default when this traverser hasn't written one yet).
                None => stream
                    .into_iter()
                    .map(|t| {
                        let v = t.sack.as_deref().cloned().unwrap_or_else(|| default.clone());
                        t.with(v)
                    })
                    .collect(),
                // `sack(op).by(proj)` — merge the projected value into the sack and
                // pass the traverser through (its value is unchanged).
                Some(op) => {
                    let by = bys.first().cloned();
                    let mut next = Vec::with_capacity(stream.len());
                    for mut t in stream {
                        let projected = match &by {
                            Some(b) => eval_by(graph, ctx, b, &t.val),
                            None => t.val.clone(),
                        };
                        let current =
                            t.sack.as_deref().cloned().unwrap_or_else(|| default.clone());
                        t.sack = Some(Box::new(apply_sack_op(*op, &current, &projected)));
                        next.push(t);
                    }
                    next
                }
            }
        }
    }
}

/// `tree()`: fold every traverser's path into one nested map, keyed level by
/// level (by the `by`-projected step values, or the raw values by default).
fn apply_tree(graph: &mut Graph, ctx: &mut Ctx, bys: &[By], stream: Vec<Trav>) -> Vec<Trav> {
    // Build a nested map from each traverser's path.
    let mut root: Vec<(GVal, GVal)> = Vec::new();
    for t in &stream {
        let keys: Vec<GVal> = t
            .path
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if bys.is_empty() {
                    v.clone()
                } else {
                    eval_by(graph, ctx, &bys[i % bys.len()], v)
                }
            })
            .collect();
        insert_tree(&mut root, &keys);
    }
    vec![Trav::root(GVal::Map(root))]
}

/// `dedup([labels]).by(...)`: keep the first traverser per distinct key — the
/// tuple of values tagged at `labels`, else the `by`-modulator tuple, else the
/// current value. A hash set on the hashable projection makes it O(n); a NaN in
/// the key is never a duplicate (NaN != NaN) so it passes straight through.
fn apply_dedupe(
    graph: &mut Graph,
    ctx: &mut Ctx,
    labels: &[String],
    bys: &[By],
    stream: Vec<Trav>,
) -> Vec<Trav> {
    // Key on: the tuple of values tagged at `labels` (`dedup('a','b')`),
    // else the tuple of `by` modulators (`dedup().by(...)`), else the
    // current value. A hash set on the hashable projection makes this
    // O(n); the old `Vec::contains` scan was O(n²).
    // Plain `dedup()` over the current value is by far the commonest form, and it
    // does not need the tuple shape: keying on the value directly drops a
    // `Vec<GVal>` for the key, a clone of the value into it, and a `Vec<DedupKey>`
    // for the projection — three allocations per traverser, 600k on a 200k
    // stream, to deduplicate what is usually a single element id.
    if labels.is_empty() && bys.is_empty() {
        let mut seen: HashSet<DedupKey> = HashSet::with_capacity(stream.len());
        let mut next = Vec::new();

        for t in stream {
            match dedup_key(&t.val) {
                // No key (a NaN inside the value) is never a duplicate, matching
                // the tuple path below.
                None => next.push(t),
                Some(k) => {
                    if seen.insert(k) {
                        next.push(t);
                    }
                }
            }
        }

        return next;
    }

    let mut seen: HashSet<Vec<DedupKey>> = HashSet::new();
    let mut next = Vec::new();
    for t in stream {
        let key: Vec<GVal> = if !labels.is_empty() {
            labels
                .iter()
                .map(|l| t.recall(l, Pop::Last).unwrap_or(GVal::Null))
                .collect()
        } else if bys.is_empty() {
            vec![t.val.clone()]
        } else {
            bys.iter()
                .map(|by| eval_by(graph, ctx, by, &t.val))
                .collect()
        };
        match key.iter().map(dedup_key).collect::<Option<Vec<DedupKey>>>() {
            // A NaN anywhere in the key is never equal to anything (NaN !=
            // NaN), so it can't be a duplicate — pass it straight through,
            // exactly as the old structural `Vec::contains` scan did.
            None => next.push(t),
            Some(dk) => {
                if seen.insert(dk) {
                    next.push(t);
                }
            }
        }
    }
    next
}

/// `order(...)`: sort the stream (Global) or, for `Scope::Local`, within each
/// traverser's value (a Map's entries / a list's elements) by the by-projected
/// keys under each by's direction.
fn apply_order(
    graph: &mut Graph,
    ctx: &mut Ctx,
    bys: &[By],
    desc: bool,
    scope: &Scope,
    stream: Vec<Trav>,
) -> Vec<Trav> {
    let bys: Vec<By> = if bys.is_empty() {
        vec![By::Identity(None)]
    } else {
        bys.to_vec()
    };

    // Compare two by-projected key vectors under the per-by direction.
    let cmp_keys = |ka: &[GVal], kb: &[GVal]| -> Ordering {
        for (i, by) in bys.iter().enumerate() {
            let dir = by
                .direction()
                .unwrap_or(if desc { Order::Desc } else { Order::Asc });
            // `cmp_or_fault` records a type fault for an incomparable pair (so
            // `try_run` still errors); fall back to the TOTAL `gcmp_total` for the
            // ordering so `sort_by` never sees a non-total comparator and panics.
            let mut o = cmp_or_fault(&ka[i], &kb[i]).unwrap_or_else(|| gcmp_total(&ka[i], &kb[i]));
            if dir == Order::Desc {
                o = o.reverse();
            }
            if o != Ordering::Equal {
                return o;
            }
        }
        Ordering::Equal
    };

    match scope {
        // Local: sort WITHIN each traverser's value — a Map's entries by
        // their VALUE (the groupCount top-N idiom; Column-parameterized
        // by(values)/by(keys) isn't modeled → local order on a Map is by
        // value), or a list's elements. A scalar has nothing to sort.
        Scope::Local => stream
            .into_iter()
            .map(|t| {
                let val = match &t.val {
                    GVal::Map(entries) => {
                        // Sort a Map's entries. `by(keys)` sorts on the entry
                        // KEY, `by(values)` (the default) on its VALUE, and a
                        // key/traversal by projects out of the value.
                        let mut es: Vec<(Vec<GVal>, (GVal, GVal))> = entries
                            .iter()
                            .map(|(k, v)| {
                                let key: Vec<GVal> = bys
                                    .iter()
                                    .map(|by| match by {
                                        By::Column(Column::Keys, _) => k.clone(),
                                        By::Column(Column::Values, _) => v.clone(),
                                        _ => eval_by(graph, ctx, by, v),
                                    })
                                    .collect();
                                (key, (k.clone(), v.clone()))
                            })
                            .collect();
                        es.sort_by(|(ka, _), (kb, _)| cmp_keys(ka, kb));
                        GVal::Map(es.into_iter().map(|(_, e)| e).collect())
                    }
                    GVal::List(items) => {
                        let mut xs: Vec<(Vec<GVal>, GVal)> = items
                            .iter()
                            .map(|x| {
                                (
                                    bys.iter().map(|by| eval_by(graph, ctx, by, x)).collect(),
                                    x.clone(),
                                )
                            })
                            .collect();
                        xs.sort_by(|(ka, _), (kb, _)| cmp_keys(ka, kb));
                        GVal::List(xs.into_iter().map(|(_, x)| x).collect())
                    }
                    _ => return t,
                };
                t.step(val)
            })
            .collect(),
        // Global: sort the traversers across the stream by their value.
        Scope::Global => {
            // Precompute sort keys (eval_by needs &mut; not usable in the comparator).
            //
            // A single-`by` fast path carrying the key as a bare `GVal` instead
            // of a one-element `Vec` measured WORSE — 8.66 -> 11.10 ms on a 50k
            // sort. The vector is an indirection, but it makes the sorted tuple
            // SMALLER (24 bytes against 40) next to a ~104-byte `Trav`, and a
            // sort moves those tuples far more often than it dereferences them.
            let mut keyed: Vec<(Vec<GVal>, Trav)> = stream
                .into_iter()
                .map(|t| {
                    (
                        bys.iter()
                            .map(|by| eval_by(graph, ctx, by, &t.val))
                            .collect(),
                        t,
                    )
                })
                .collect();
            keyed.sort_by(|(ka, _), (kb, _)| cmp_keys(ka, kb));
            keyed.into_iter().map(|(_, t)| t).collect()
        }
    }
}

/// `sample(k)`: a pseudo-random sample (partial Fisher-Yates), NOT a prefix. The
/// fixed-seed Mulberry32 makes it reproducible and byte-identical with the TS
/// engine's `sampleStep`, which runs the same shuffle.
fn apply_sample(n: usize, stream: Vec<Trav>) -> Vec<Trav> {
    let mut buf = stream;
    let len = buf.len();
    let k = n.min(len);
    let mut rng = Mulberry32::new(SAMPLE_SEED);
    for i in 0..k {
        let j = i + (rng.next_f64() * (len - i) as f64) as usize;
        buf.swap(i, j);
    }
    buf.truncate(k);
    buf
}

/// Merge a projected value into the sack: `newSack = op(currentSack, projected)`.
/// `Assign` replaces; the folds require both operands numeric (else `Null`, as the
/// TS engine yields) and use raw `<=`/`>=` for min/max so NaN resolves identically
/// on both engines (NaN comparisons are false in Rust and JS alike).
fn apply_sack_op(op: SackOp, current: &GVal, projected: &GVal) -> GVal {
    let num = |v: &GVal| match v {
        GVal::Num(n) => Some(*n),
        _ => None,
    };
    match op {
        SackOp::Assign => projected.clone(),
        _ => match (num(current), num(projected)) {
            (Some(a), Some(b)) => GVal::Num(match op {
                SackOp::Sum => a + b,
                SackOp::Mult => a * b,
                SackOp::Min => {
                    if a <= b {
                        a
                    } else {
                        b
                    }
                }
                SackOp::Max => {
                    if a >= b {
                        a
                    } else {
                        b
                    }
                }
                SackOp::Assign => unreachable!(),
            }),
            _ => GVal::Null,
        },
    }
}

/// `(out, inn)` flags as the shared [`crate::seek::Dir`].
fn seek_dir(out: bool, inn: bool) -> crate::seek::Dir {
    match (out, inn) {
        (true, true) => crate::seek::Dir::Both,
        (false, true) => crate::seek::Dir::In,
        _ => crate::seek::Dir::Out,
    }
}

/// Label names as edge-type ids: `Some(ids)` to match those types, `Some(empty)`
/// to match ANY, `None` to match nothing.
///
/// An unknown label contributes no edges but does not void the others —
/// `both('KNOWS','CREATED','BLAH')` still walks KNOWS and CREATED. Only when
/// labels were given and NONE resolve does the step match nothing, which is
/// distinct from no labels at all matching everything. Conflating those two made
/// `[:NONEXISTENT]` match every edge, three times, in three places.
fn resolve_etypes(graph: &Graph, labels: &[String]) -> Option<Vec<u32>> {
    if labels.is_empty() {
        return Some(Vec::new()); // no filter: any type
    }

    let ids: Vec<u32> = labels.iter().filter_map(|l| graph.etype.get(l)).collect();

    (!ids.is_empty()).then_some(ids)
}

fn dir_flags(step: &Step) -> (bool, bool) {
    match step {
        Step::Out(_) | Step::OutE(_) => (true, false),
        Step::In(_) | Step::InE(_) => (false, true),
        _ => (true, true),
    }
}

/// Resolve an `addE` endpoint to a vertex id.
fn resolve_endpoint(graph: &mut Graph, ctx: &mut Ctx, ep: &Endpoint, t: &Trav) -> Option<u32> {
    let v = match ep {
        Endpoint::Current => t.val.clone(),
        Endpoint::Tag(label) => t.recall(label, Pop::Last)?,
        Endpoint::Plan(plan) => sub_vals(graph, ctx, plan, t).into_iter().next()?,
    };
    match v {
        GVal::Node(i) => Some(i),
        _ => None,
    }
}

/// The `value` field of a `{key, value}` property map (for `value`/`hasValue`).
fn prop_value_field(v: &GVal) -> Option<GVal> {
    match v {
        GVal::Property(p) => Some(p.value.clone()),
        GVal::Map(entries) => entries
            .iter()
            .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == "value"))
            .map(|(_, x)| x.clone()),
        _ => None,
    }
}
fn prop_key_field(v: &GVal) -> Option<GVal> {
    match v {
        GVal::Property(p) => Some(GVal::Str(p.key.clone())),
        GVal::Map(entries) => entries
            .iter()
            .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == "key"))
            .map(|(_, x)| x.clone()),
        _ => None,
    }
}

/// Substitute a comparison predicate's RHS with a resolved value (`where(key,pred)`).
fn substitute_rhs(p: &P, v: GVal) -> P {
    match p {
        P::Eq(_) => P::Eq(v),
        P::Neq(_) => P::Neq(v),
        P::Gt(_) => P::Gt(v),
        P::Gte(_) => P::Gte(v),
        P::Lt(_) => P::Lt(v),
        P::Lte(_) => P::Lte(v),
        other => other.clone(),
    }
}

fn slice_local(v: &GVal, start: usize, end: usize) -> GVal {
    let e = local_elems(v);
    let s = start.min(e.len());
    let en = end.min(e.len());
    GVal::list(if s < en {
        e[s..en].to_vec()
    } else {
        Vec::new()
    })
}

fn map_step(stream: Vec<Trav>, f: impl Fn(&Trav) -> Vec<GVal>) -> Vec<Trav> {
    let mut next = Vec::new();
    for t in &stream {
        for v in f(t) {
            next.push(t.with(v));
        }
    }
    next
}

fn has_dup(path: &[GVal]) -> bool {
    // O(path) via a DedupKey set, replacing the O(path²) pairwise `==` scan (paths
    // grow under long repeat()). A NaN-bearing value has no DedupKey and — like the
    // `==` scan it replaces — can never be a duplicate (NaN != NaN), so skip it.
    let mut seen: HashSet<DedupKey> = HashSet::with_capacity(path.len());
    for v in path {
        if let Some(dk) = dedup_key(v) {
            if !seen.insert(dk) {
                return true;
            }
        }
    }
    false
}

fn fold_num(stream: Vec<Trav>, f: impl Fn(&[f64]) -> f64) -> Vec<Trav> {
    let ns: Vec<f64> = stream.iter().filter_map(|t| strict_num(&t.val)).collect();
    if ns.is_empty() {
        // No numeric values (empty stream or all null) → a single null result,
        // matching the TS engine's `sum`/`mean`.
        vec![Trav::root(GVal::Null)]
    } else {
        vec![Trav::root(GVal::Num(f(&ns)))]
    }
}

fn local_num(v: &GVal, f: impl Fn(&[f64]) -> f64) -> GVal {
    let ns: Vec<f64> = local_elems(v).iter().filter_map(strict_num).collect();
    if ns.is_empty() {
        GVal::Null
    } else {
        GVal::Num(f(&ns))
    }
}

fn fold_extreme(stream: Vec<Trav>, want: Ordering) -> Vec<Trav> {
    let mut best: Option<GVal> = None;
    for t in stream {
        if matches!(t.val, GVal::Null) {
            continue; // TinkerPop min/max ignore nulls
        }
        best = Some(match best {
            None => t.val,
            Some(b) => {
                if cmp_or_fault(&t.val, &b) == Some(want) {
                    t.val
                } else {
                    b
                }
            }
        });
    }
    // No non-null value (empty or all-null) → a single null, matching TS.
    vec![Trav::root(best.unwrap_or(GVal::Null))]
}

fn local_extreme(v: &GVal, want: Ordering) -> GVal {
    let mut best: Option<GVal> = None;
    for e in local_elems(v) {
        if matches!(e, GVal::Null) {
            continue; // ignore nulls, like min/max
        }
        best = Some(match best {
            None => e,
            Some(b) => {
                if cmp_or_fault(&e, &b) == Some(want) {
                    e
                } else {
                    b
                }
            }
        });
    }
    best.unwrap_or(GVal::Null)
}

/// Insert a key chain into a nested tree map (for `tree()`).
fn insert_tree(node: &mut Vec<(GVal, GVal)>, keys: &[GVal]) {
    let Some((head, rest)) = keys.split_first() else {
        return;
    };
    let child = match node.iter_mut().find(|(k, _)| k == head) {
        Some((_, GVal::Map(m))) => m,
        Some(_) => return,
        None => {
            node.push((head.clone(), GVal::Map(Vec::new())));
            match &mut node.last_mut().unwrap().1 {
                GVal::Map(m) => m,
                _ => unreachable!(),
            }
        }
    };
    insert_tree(child, rest);
}

/// The `repeat()` modulators, bundled so `run_repeat` stays under the arg limit:
/// how many times, the `until`/`emit` sub-traversals, and whether each is checked
/// before the body (do-while / emit-before-step) vs after.
struct RepeatMods<'a> {
    times: Option<usize>,
    until: Option<&'a Traversal>,
    until_before: bool,
    emit: Option<&'a Traversal>,
    emit_before: bool,
}

/// `repeat(body)` with `times` / `until` / `emit` modulators.
fn run_repeat(
    graph: &mut Graph,
    ctx: &mut Ctx,
    stream: &[Trav],
    body: &Traversal,
    mods: RepeatMods,
) -> Vec<Trav> {
    let RepeatMods {
        times,
        until,
        until_before,
        emit,
        emit_before,
    } = mods;
    // Default iteration bound for a `repeat()` with no `times()` — must match the TS
    // engine's cap (packages/gremlin/.../iteration.ts) so an unbounded/deep repeat
    // stops at the same depth on both. (This is the per-count default, separate from
    // the REPEAT_BUDGET traverser-count safety.)
    const CAP: usize = 100;
    let emit_matches = |graph: &mut Graph, ctx: &mut Ctx, t: &Trav, e: &Traversal| {
        e.steps.is_empty() || sub_nonempty(graph, ctx, e, t)
    };

    // loops() counts from 1 inside the first body pass: increment on entry
    // (the input frontier) and again on each body output, matching TS so that
    // `until`/`emit` predicates over loops() agree across engines.
    if until.is_none() && emit.is_none() {
        let n = times.unwrap_or(CAP);
        let mut current: Vec<Trav> = stream.iter().cloned().map(inc_loops).collect();
        let mut work: u64 = 0;
        for _ in 0..n {
            if current.is_empty() {
                break;
            }
            current = run_steps(graph, ctx, &body.steps, current)
                .into_iter()
                .map(inc_loops)
                .collect();
            work += current.len() as u64;
            if work > REPEAT_BUDGET {
                ctx.over_budget = true;
                break;
            }
        }
        return current;
    }

    let mut out: Vec<Trav> = Vec::new();
    let mut current: Vec<Trav> = stream.iter().cloned().map(inc_loops).collect();
    let max = times.unwrap_or(CAP);
    let mut work: u64 = 0;
    for _ in 0..max {
        if current.is_empty() {
            break;
        }
        // Pre-form emit (TinkerPop's `emit(...).repeat(body)`): emit the current
        // frontier before applying the body, so every level start is emitted
        // (level 0 = the input traverser), not just the initial frontier.
        if emit_before {
            if let Some(e) = emit {
                for t in &current {
                    if emit_matches(graph, ctx, t, e) {
                        out.push(t.clone());
                    }
                }
            }
        }
        // while-do: pre-form `until(cond).repeat(body)` checks BEFORE the body —
        // a satisfier exits without ever running it.
        let advancing = if until_before {
            let mut adv = Vec::new();
            for t in std::mem::take(&mut current) {
                match until {
                    Some(u) if sub_nonempty(graph, ctx, u, &t) => out.push(t),
                    _ => adv.push(t),
                }
            }
            adv
        } else {
            std::mem::take(&mut current)
        };

        let stepped: Vec<Trav> = run_steps(graph, ctx, &body.steps, advancing)
            .into_iter()
            .map(inc_loops)
            .collect();
        if !emit_before {
            if let Some(e) = emit {
                for t in &stepped {
                    if emit_matches(graph, ctx, t, e) {
                        out.push(t.clone());
                    }
                }
            }
        }
        work += stepped.len() as u64;
        if work > REPEAT_BUDGET {
            ctx.over_budget = true;
            break;
        }

        // do-while: post-form `repeat(body).until(cond)` checks AFTER the body —
        // a satisfier exits; the rest loop on.
        if until.is_some() && !until_before {
            let mut cont = Vec::new();
            for t in stepped {
                match until {
                    Some(u) if sub_nonempty(graph, ctx, u, &t) => out.push(t),
                    _ => cont.push(t),
                }
            }
            current = cont;
        } else {
            current = stepped;
        }
    }
    // Pre-emit form yields the final frontier too (it never got a pre-emit pass).
    if emit_before && until.is_none() {
        out.extend(current);
    }
    out
}

fn inc_loops(mut t: Trav) -> Trav {
    t.loops += 1;
    t
}

#[cfg(test)]
mod dedup_key_tests {
    use std::sync::Arc;

    use super::{dedup_key, GVal};

    // The hashed dedup key must mirror `GVal`'s structural `PartialEq` exactly,
    // with the f64 edge cases the hash set can't express structurally.
    #[test]
    fn float_and_nan_semantics() {
        // -0.0 and +0.0 are `==`, so they share a key (dedup together).
        assert_eq!(dedup_key(&GVal::Num(0.0)), dedup_key(&GVal::Num(-0.0)));
        // distinct numbers get distinct keys.
        assert_ne!(dedup_key(&GVal::Num(1.0)), dedup_key(&GVal::Num(2.0)));
        // NaN != NaN, so a NaN is never a duplicate → no key (pass-through).
        assert!(dedup_key(&GVal::Num(f64::NAN)).is_none());
        // a NaN nested in a list/map makes the whole key un-hashable too.
        assert!(dedup_key(&GVal::list(vec![GVal::Num(1.0), GVal::Num(f64::NAN)])).is_none());
        assert!(dedup_key(&GVal::Map(vec![(
            GVal::Str(Arc::from("k")),
            GVal::Num(f64::NAN)
        )]))
        .is_none());
    }

    #[test]
    fn element_and_value_keys() {
        assert_eq!(dedup_key(&GVal::Node(7)), dedup_key(&GVal::Node(7)));
        assert_ne!(dedup_key(&GVal::Node(7)), dedup_key(&GVal::Node(8)));
        // same id, different element kind → different key (a vertex isn't an edge).
        assert_ne!(dedup_key(&GVal::Node(7)), dedup_key(&GVal::Edge(7)));
        assert_eq!(
            dedup_key(&GVal::Str(Arc::from("x"))),
            dedup_key(&GVal::Str(Arc::from("x")))
        );
        assert_ne!(
            dedup_key(&GVal::Str(Arc::from("x"))),
            dedup_key(&GVal::Str(Arc::from("y")))
        );
        // distinct variants never collide.
        assert_ne!(dedup_key(&GVal::Bool(true)), dedup_key(&GVal::Num(1.0)));
        assert_ne!(dedup_key(&GVal::Null), dedup_key(&GVal::Bool(false)));
        // nested structure keys element-wise (incl. the -0.0/+0.0 collapse).
        assert_eq!(
            dedup_key(&GVal::list(vec![GVal::Num(-0.0), GVal::Node(1)])),
            dedup_key(&GVal::list(vec![GVal::Num(0.0), GVal::Node(1)])),
        );
    }
}
