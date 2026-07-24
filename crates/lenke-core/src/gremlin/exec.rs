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
use crate::graph::{Graph, IdxKey, RangeBound, Value};
use crate::jsonfmt::{push_json_str, push_num};

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
    Some(match v {
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
        GVal::Vertex(id) => DedupKey::Vertex(*id),
        GVal::Edge(id) => DedupKey::Edge(*id),
        GVal::List(xs) => DedupKey::List(xs.iter().map(dedup_key).collect::<Option<_>>()?),
        GVal::Map(kvs) => DedupKey::Map(
            kvs.iter()
                .map(|(k, val)| Some((dedup_key(k)?, dedup_key(val)?)))
                .collect::<Option<_>>()?,
        ),
        // Owner-agnostic (like PartialEq / the TS engine): dedup a property
        // element by its key+value.
        GVal::Property { key, value, .. } => {
            DedupKey::List(vec![DedupKey::Str(key.clone()), dedup_key(value)?])
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
    tags: Vec<(String, Vec<GVal>)>,
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
            tags: Vec::new(),
            loops: 0,
            sack: None,
        }
    }
    /// A successor that moved to `val` (extends path, keeps tags/loops/sack).
    fn step(&self, val: GVal) -> Self {
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
            Pop::All => Some(GVal::List(list.clone())),
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
    /// unresolvable endpoint). Surfaced by [`try_run`]; ignored by [`run`].
    fault: Option<(crate::error_codes::ErrorCode, &'static str)>,
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
                       // Index seeding: `V().has(key, pred)` on an indexed key seeds from the
                       // property index (skipping the full label scan + the now-satisfied `has`).
    let (seed, start) = match index_seed(graph, &t.steps) {
        Some(s) => (s, 2),
        None => (Vec::new(), 0),
    };
    run_steps(graph, ctx, &t.steps[start..], seed)
        .into_iter()
        .map(|t| t.val)
        .collect()
}

fn gval_to_idxkey(v: &GVal) -> Option<IdxKey> {
    match v {
        GVal::Str(s) => Some(IdxKey::Str(s.clone())),
        GVal::Num(n) => Some(IdxKey::Num(*n)),
        GVal::Bool(b) => Some(IdxKey::Bool(*b)),
        _ => None,
    }
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

/// If the plan opens with `V().has(key, pred)` / `E().has(key, pred)` on an
/// indexed key and `pred` is index-seekable (eq / within / range / startsWith),
/// return the seeded elements (the `has` is then fully satisfied by the index).
/// `None` ⇒ fall back to scan.
fn index_seed(graph: &Graph, steps: &[Step]) -> Option<Vec<Trav>> {
    let (key, pred, is_edge) = match steps {
        [Step::V(ids), Step::Has(k, p), ..] if ids.is_empty() && graph.vertex_indexed(k) => {
            (k, p, false)
        }
        [Step::E(ids), Step::Has(k, p), ..] if ids.is_empty() && graph.edge_indexed(k) => {
            (k, p, true)
        }
        _ => return None,
    };
    let eq = |k: &IdxKey| {
        if is_edge {
            graph.edges_by_prop(key, k)
        } else {
            graph.vertices_by_prop(key, k)
        }
    };
    let rng = |b: RangeBound| {
        if is_edge {
            graph.edges_by_prop_range(key, &b)
        } else {
            graph.vertices_by_prop_range(key, &b)
        }
    };
    let ids: Vec<u32> = match pred {
        P::Eq(v) => eq(&gval_to_idxkey(v)?)?.to_vec(),
        P::Within(vs) => {
            let mut out = Vec::new();
            for v in vs {
                if let Some(k) = gval_to_idxkey(v) {
                    if let Some(s) = eq(&k) {
                        out.extend_from_slice(s);
                    }
                }
            }
            out
        }
        P::Gt(v) => rng(RangeBound {
            gt: gval_to_idxkey(v),
            ..Default::default()
        })?,
        P::Gte(v) => rng(RangeBound {
            gte: gval_to_idxkey(v),
            ..Default::default()
        })?,
        P::Lt(v) => rng(RangeBound {
            lt: gval_to_idxkey(v),
            ..Default::default()
        })?,
        P::Lte(v) => rng(RangeBound {
            lte: gval_to_idxkey(v),
            ..Default::default()
        })?,
        P::Between(lo, hi) => rng(RangeBound {
            gte: gval_to_idxkey(lo),
            lt: gval_to_idxkey(hi),
            ..Default::default()
        })?,
        P::Inside(lo, hi) => rng(RangeBound {
            gt: gval_to_idxkey(lo),
            lt: gval_to_idxkey(hi),
            ..Default::default()
        })?,
        P::Outside(lo, hi) => {
            let mut out = rng(RangeBound {
                lt: gval_to_idxkey(lo),
                ..Default::default()
            })?;
            out.extend(rng(RangeBound {
                gt: gval_to_idxkey(hi),
                ..Default::default()
            })?);
            out
        }
        P::StartsWith(prefix) => {
            let lo = Some(IdxKey::Str(prefix.as_str().into()));
            let hi = prefix_upper(prefix).map(|u| IdxKey::Str(u.as_str().into()));
            rng(RangeBound {
                gte: lo,
                lt: hi,
                ..Default::default()
            })?
        }
        _ => return None,
    };
    let mk = |id: u32| {
        if is_edge {
            GVal::Edge(id)
        } else {
            GVal::Vertex(id)
        }
    };
    Some(ids.into_iter().map(|id| Trav::root(mk(id))).collect())
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
        GVal::Vertex(i) => push_result_value(out, &crate::gql::eval::node_result_value(graph, *i)),
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
        GVal::Property { key, value, .. } => {
            out.push_str("{\"key\":");
            push_json_str(out, key);
            out.push_str(",\"value\":");
            write_gval(out, graph, value);
            out.push('}');
        }
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
    match nt.tags.iter_mut().find(|(l, _)| l == key) {
        Some((_, list)) => *list = vec![val],
        None => nt.tags.push((key.to_string(), vec![val])),
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
            for (_, n) in adj_in_label_order(graph, v, out, inn, &[]) {
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
            "graph algorithm step failed",
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
                !run_steps(graph, ctx, &plan.steps, vec![Trav::root(GVal::Vertex(v))]).is_empty()
            })
            .collect()
    });
    let mut next = Vec::new();
    for t in &stream {
        if let GVal::Vertex(src) = t.val {
            for path in shortest_paths_from(graph, src, targets.as_ref(), out, inn) {
                next.push(t.with(GVal::List(path.into_iter().map(GVal::Vertex).collect())));
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
    match v {
        Value::Null => GVal::Null,
        Value::Bool(b) => GVal::Bool(b),
        Value::Num(n) => GVal::Num(n),
        Value::Str(s) => GVal::Str(s),
        Value::Temporal(t) => GVal::Temporal(t),
        Value::List(items) => GVal::List(items.into_iter().map(value_to_gval).collect()),
        // Map is a GQL-result-only value; it never reaches the Gremlin value path.
        Value::Map(_) => {
            unreachable!("Value::Map is a GQL query-result value, not a Gremlin value")
        }
    }
}

fn gval_to_value(v: &GVal) -> Value {
    match v {
        GVal::Null => Value::Null,
        GVal::Bool(b) => Value::Bool(*b),
        GVal::Num(n) => Value::Num(*n),
        GVal::Str(s) => Value::Str(s.clone()),
        GVal::Temporal(t) => Value::Temporal(*t),
        GVal::List(items) => Value::List(items.iter().map(gval_to_value).collect()),
        _ => Value::Null,
    }
}

fn prop(graph: &Graph, v: &GVal, key: &str) -> GVal {
    match v {
        GVal::Vertex(i) => value_to_gval(graph.props.value(*i as usize, key, &graph.strs)),
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
    let gv = GVal::Vertex(v);
    let labels: Vec<GVal> = graph
        .vertex_labels(v)
        .iter()
        .map(|&l| GVal::Str(graph.labels.arc(l)))
        .collect();
    GVal::Map(vec![
        (GVal::Str(Arc::from("id")), GVal::Str(graph.vid.arc(v))),
        (GVal::Str(Arc::from("labels")), GVal::List(labels)),
        (
            GVal::Str(Arc::from("properties")),
            element_props_map(graph, &gv),
        ),
    ])
}

/// A self-describing edge record: `{ id, label, outV, inV, properties }`.
fn subgraph_edge(graph: &Graph, e: u32) -> GVal {
    let ge = GVal::Edge(e);
    let outv = GVal::Vertex(graph.e_src[e as usize]);
    let inv = GVal::Vertex(graph.e_dst[e as usize]);
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

fn present_keys(graph: &Graph, v: &GVal) -> Vec<String> {
    // A property element (from `properties()`) exposes its own single key, so
    // `hasKey`/`hasNot` filter a property stream by the property's key field.
    if let Some(GVal::Str(k)) = prop_key_field(v) {
        return vec![k.to_string()];
    }
    let (store, idx) = match v {
        GVal::Vertex(i) => (&graph.props, *i as usize),
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
        GVal::Vertex(i) => graph.props.is_present(*i as usize, key),
        GVal::Edge(e) => graph.edge_props.is_present(*e as usize, key),
        _ => false,
    }
}

fn elem_id(graph: &Graph, v: &GVal) -> GVal {
    match v {
        GVal::Vertex(i) => GVal::Str(graph.vid.arc(*i)),
        // Every edge has an id (assigned external id, else canonical `e{index}`).
        GVal::Edge(e) => GVal::Str(Arc::from(graph.edge_id(*e).as_ref())),
        other => other.clone(),
    }
}

fn elem_label(graph: &Graph, v: &GVal) -> GVal {
    match v {
        GVal::Vertex(i) => match graph.vertex_labels(*i).first() {
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
/// ≤1-ULP glibc-`powf`-vs-V8-`pow` difference on some inputs (see
/// `docs/dogfood/findings/round15.md` — a documented won't-fix). Arity mismatch
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
            GVal::Vertex(_) | GVal::Edge(_) => prop(graph, value, key),
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
                GVal::List(outs)
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
        GVal::List(items) => items.clone(),
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
                verts.into_iter().map(|v| Trav::root(GVal::Vertex(v))).collect()
            } else {
                stream.iter().flat_map(|t| verts.iter().map(move |&v| t.step(GVal::Vertex(v)))).collect()
            }
        }
        Step::E(ids) => {
            // Match an external edge id (e.g. `E('7')`, like `V('1')`), falling
            // back to the synthetic `e{index}` form when no external id was set.
            let edges: Vec<u32> = (0..graph.e_src.len() as u32)
                .filter(|&e| graph.is_edge_live(e))
                .filter(|&e| {
                    // `edge_by_id` resolves both an assigned id and the canonical
                    // `e{index}` form, so no separate synthetic fallback is needed.
                    ids.is_empty() || ids.iter().any(|i| graph.edge_by_id(i) == Some(e))
                })
                .collect();
            if stream.is_empty() {
                edges.into_iter().map(|e| Trav::root(GVal::Edge(e))).collect()
            } else {
                stream.iter().flat_map(|t| edges.iter().map(move |&e| t.step(GVal::Edge(e)))).collect()
            }
        }

        // --- vertex → vertex (multi-label emits in label-arg order) ---
        Step::Out(labels) | Step::In(labels) | Step::Both(labels) => {
            let (out, inn) = dir_flags(step);
            let mut next = Vec::new();
            for t in &stream {
                if let GVal::Vertex(v) = t.val {
                    for a in adj_in_label_order(graph, v, out, inn, labels) {
                        next.push(t.step(GVal::Vertex(a.1)));
                    }
                }
            }
            next
        }

        // --- vertex → edge ---
        Step::OutE(labels) | Step::InE(labels) | Step::BothE(labels) => {
            let (out, inn) = dir_flags(step);
            let mut next = Vec::new();
            for t in &stream {
                if let GVal::Vertex(v) = t.val {
                    for a in adj_in_label_order(graph, v, out, inn, labels) {
                        next.push(t.step(GVal::Edge(a.0)));
                    }
                }
            }
            next
        }

        // --- edge → vertex ---
        Step::OutV => map_step(stream, |t| match t.val {
            GVal::Edge(e) => vec![GVal::Vertex(graph.e_src[e as usize])],
            _ => vec![],
        }),
        Step::InV => map_step(stream, |t| match t.val {
            GVal::Edge(e) => vec![GVal::Vertex(graph.e_dst[e as usize])],
            _ => vec![],
        }),
        Step::BothV => map_step(stream, |t| match t.val {
            GVal::Edge(e) => vec![GVal::Vertex(graph.e_src[e as usize]), GVal::Vertex(graph.e_dst[e as usize])],
            _ => vec![],
        }),
        Step::OtherV => {
            let mut next = Vec::new();
            for t in &stream {
                if let GVal::Edge(e) = t.val {
                    let (src, dst) = (graph.e_src[e as usize], graph.e_dst[e as usize]);
                    let from = t.path.iter().rev().nth(1).and_then(|g| match g {
                        GVal::Vertex(v) => Some(*v),
                        _ => None,
                    });
                    next.push(t.step(GVal::Vertex(if from == Some(src) { dst } else { src })));
                }
            }
            next
        }

        // --- filters ---
        Step::Has(key, pred) => stream.into_iter().filter(|t| p_matches(pred, &prop(graph, &t.val, key))).collect(),
        Step::HasLabel(labels) => stream
            .into_iter()
            .filter(|t| matches!(elem_label(graph, &t.val), GVal::Str(ref s) if labels.iter().any(|l| l == s.as_ref())))
            .collect(),
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
        Step::Dedupe { labels, bys } => {
            // Key on: the tuple of values tagged at `labels` (`dedup('a','b')`),
            // else the tuple of `by` modulators (`dedup().by(...)`), else the
            // current value. A hash set on the hashable projection makes this
            // O(n); the old `Vec::contains` scan was O(n²).
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
                    bys.iter().map(|by| eval_by(graph, ctx, by, &t.val)).collect()
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

        // --- projection ---
        Step::Values(keys) => {
            let mut next = Vec::new();
            for t in &stream {
                let ks = if keys.is_empty() { present_keys(graph, &t.val) } else { keys.clone() };
                for k in ks {
                    // Gate on PRESENCE, not value != Null: a present null yields
                    // a `Null` here; only an absent key is skipped.
                    if prop_present(graph, &t.val, &k) {
                        next.push(t.step(prop(graph, &t.val, &k)));
                    }
                }
            }
            next
        }
        Step::ValueMap(keys) => map_step(stream, |t| {
            let ks = if keys.is_empty() { present_keys(graph, &t.val) } else { keys.clone() };
            let entries = ks
                .into_iter()
                .filter(|k| prop_present(graph, &t.val, k))
                .map(|k| (GVal::Str(Arc::from(k.as_str())), prop(graph, &t.val, &k)))
                .collect();
            vec![GVal::Map(entries)]
        }),
        Step::PropertyMap(keys) => map_step(stream, |t| {
            let ks = if keys.is_empty() { present_keys(graph, &t.val) } else { keys.clone() };
            let entries = ks
                .into_iter()
                .filter(|k| prop_present(graph, &t.val, k))
                .map(|k| (GVal::Str(Arc::from(k.as_str())), GVal::List(vec![prop(graph, &t.val, &k)])))
                .collect();
            vec![GVal::Map(entries)]
        }),
        Step::ElementMap(keys) => map_step(stream, |t| {
            if !matches!(t.val, GVal::Vertex(_) | GVal::Edge(_)) {
                return vec![];
            }
            let mut entries = vec![
                (GVal::Str(Arc::from("id")), elem_id(graph, &t.val)),
                (GVal::Str(Arc::from("label")), elem_label(graph, &t.val)),
            ];
            if let GVal::Edge(e) = t.val {
                let inv = GVal::Vertex(graph.e_dst[e as usize]);
                let outv = GVal::Vertex(graph.e_src[e as usize]);
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
                        next.push(t.step(GVal::Property {
                            owner: Box::new(t.val.clone()),
                            key: Arc::from(k.as_str()),
                            value: Box::new(v),
                        }));
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
                t.with(GVal::List(projected))
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
        Step::Tree(bys) => {
            // Build a nested map from each traverser's path.
            let mut root: Vec<(GVal, GVal)> = Vec::new();
            for t in &stream {
                let keys: Vec<GVal> = t
                    .path
                    .iter()
                    .enumerate()
                    .map(|(i, v)| if bys.is_empty() { v.clone() } else { eval_by(graph, ctx, &bys[i % bys.len()], v) })
                    .collect();
                insert_tree(&mut root, &keys);
            }
            vec![Trav::root(GVal::Map(root))]
        }

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
            vec![GVal::List(e[start..].to_vec())]
        }),
        Step::Sample(n) => {
            // A pseudo-random sample (partial Fisher-Yates), NOT a prefix. The
            // fixed-seed Mulberry32 makes it reproducible and byte-identical with
            // the TS engine's `sampleStep`, which runs the same shuffle.
            let mut buf = stream;
            let len = buf.len();
            let k = (*n).min(len);
            let mut rng = Mulberry32::new(SAMPLE_SEED);
            for i in 0..k {
                let j = i + (rng.next_f64() * (len - i) as f64) as usize;
                buf.swap(i, j);
            }
            buf.truncate(k);
            buf
        }

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
        Step::Order(bys, desc, scope) => {
            let bys: Vec<By> = if bys.is_empty() { vec![By::Identity(None)] } else { bys.clone() };

            // Compare two by-projected key vectors under the per-by direction.
            let cmp_keys = |ka: &[GVal], kb: &[GVal]| -> Ordering {
                for (i, by) in bys.iter().enumerate() {
                    let dir = by.direction().unwrap_or(if *desc { Order::Desc } else { Order::Asc });
                    let mut o = cmp_or_fault(&ka[i], &kb[i]).unwrap_or(Ordering::Equal);
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
                                    .map(|x| (bys.iter().map(|by| eval_by(graph, ctx, by, x)).collect(), x.clone()))
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
                    let mut keyed: Vec<(Vec<GVal>, Trav)> = stream
                        .into_iter()
                        .map(|t| (bys.iter().map(|by| eval_by(graph, ctx, by, &t.val)).collect(), t))
                        .collect();
                    keyed.sort_by(|(ka, _), (kb, _)| cmp_keys(ka, kb));
                    keyed.into_iter().map(|(_, t)| t).collect()
                }
            }
        }
        Step::Group(bys) => {
            let key_by = bys.first().cloned().unwrap_or(By::Identity(None));
            let val_by = bys.get(1).cloned().unwrap_or(By::Identity(None));
            // Bucket the group's MEMBERS (traversers), keeping key + insertion
            // order, so a reducing value-by can fold over each group as a barrier.
            let mut buckets: Vec<(GVal, Vec<Trav>)> = Vec::new();
            for t in stream {
                let key = eval_by(graph, ctx, &key_by, &t.val);
                match buckets.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, members)) => members.push(t),
                    None => buckets.push((key, vec![t])),
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
            for t in &stream {
                let key = eval_by(graph, ctx, &by, &t.val);
                match entries.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, n)) => *n += 1.0,
                    None => entries.push((key, 1.0)),
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
                vec![Trav::root(GVal::List(ctx.side.get(key).cloned().unwrap_or_default()))]
            }
        }
        Step::Barrier => stream,
        Step::Repeat { body, times, until, until_before, emit, emit_before } => run_repeat(graph, ctx, &stream, body, *times, until.as_deref(), *until_before, emit.as_deref(), *emit_before),

        // --- tagging / select ---
        Step::As(label) => stream
            .into_iter()
            .map(|mut t| {
                let val = t.val.clone();
                match t.tags.iter_mut().find(|(l, _)| l == label) {
                    Some((_, list)) => list.push(val),
                    None => t.tags.push((label.clone(), vec![val])),
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
                        for it in items {
                            next.push(t.step(it.clone()));
                        }
                    }
                    other => next.push(t.step(other.clone())),
                }
            }
            next
        }
        Step::Index => stream.iter().enumerate().map(|(i, t)| t.with(GVal::List(vec![t.val.clone(), GVal::Num(i as f64)]))).collect(),
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
            if !stream.is_empty() {
                panic!("{}", msg.clone().unwrap_or_else(|| "fail() reached".to_string()));
            }
            stream
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
                    "addV(): a label must be non-empty and cannot contain '::'",
                ));
                return Vec::new();
            }
            // As a source (`g.addV()`), create one even with no incoming traverser.
            let base = if stream.is_empty() { vec![Trav::root(GVal::Null)] } else { stream };
            base.iter().map(|t| t.with(GVal::Vertex(graph.add_vertex(&labels, vec![])))).collect()
        }
        Step::AddE { label, from, to } => {
            if crate::graph::validate_label(label).is_err() {
                ctx.fault.get_or_insert((
                    crate::error_codes::ErrorCode::InvalidValue,
                    "addE(): a label must be non-empty and cannot contain '::'",
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
                        "addE(): could not resolve endpoint vertices",
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
                "property(): a key must be non-empty",
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
                    GVal::Vertex(i) => Some((true, *i)),
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
                    GVal::Vertex(i) => {
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
                    GVal::Property { owner, key, .. } => match **owner {
                        GVal::Vertex(i) => graph.remove_vertex_prop(i, key),
                        GVal::Edge(e) => graph.remove_edge_prop(e, key),
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
                    "sack() requires a preceding withSack()",
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

/// Collect a vertex's adjacency as `(eidx, nbr)`. With labels, emit per label in
/// argument order (Gremlin `out('A','B')` yields all A-edges then all B-edges);
/// without, adjacency order (out then in for `both`), deduped across both for
/// `both` with no labels is not required (TinkerPop both yields each edge once
/// per direction — matches iterating out then in).
fn adj_in_label_order(
    graph: &Graph,
    v: u32,
    out: bool,
    inn: bool,
    labels: &[String],
) -> Vec<(u32, u32)> {
    let outs: Vec<(u32, u32, u32)> = if out {
        graph.out_adj(v).map(|a| (a.eidx, a.nbr, a.etype)).collect()
    } else {
        Vec::new()
    };
    let ins: Vec<(u32, u32, u32)> = if inn {
        graph.in_adj(v).map(|a| (a.eidx, a.nbr, a.etype)).collect()
    } else {
        Vec::new()
    };
    let collect_dir = |adjs: &[(u32, u32, u32)], dst: &mut Vec<(u32, u32)>| {
        if labels.is_empty() {
            dst.extend(adjs.iter().map(|a| (a.0, a.1)));
        } else {
            for lbl in labels {
                if let Some(id) = graph.etype.get(lbl) {
                    dst.extend(adjs.iter().filter(|a| a.2 == id).map(|a| (a.0, a.1)));
                }
            }
        }
    };
    let mut res = Vec::new();
    collect_dir(&outs, &mut res);
    collect_dir(&ins, &mut res);
    res
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
        GVal::Vertex(i) => Some(i),
        _ => None,
    }
}

/// The `value` field of a `{key, value}` property map (for `value`/`hasValue`).
fn prop_value_field(v: &GVal) -> Option<GVal> {
    match v {
        GVal::Property { value, .. } => Some((**value).clone()),
        GVal::Map(entries) => entries
            .iter()
            .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == "value"))
            .map(|(_, x)| x.clone()),
        _ => None,
    }
}
fn prop_key_field(v: &GVal) -> Option<GVal> {
    match v {
        GVal::Property { key, .. } => Some(GVal::Str(key.clone())),
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
    GVal::List(if s < en {
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
    for i in 0..path.len() {
        for j in (i + 1)..path.len() {
            if path[i] == path[j] {
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

/// `repeat(body)` with `times` / `until` / `emit` modulators.
#[allow(clippy::too_many_arguments)]
fn run_repeat(
    graph: &mut Graph,
    ctx: &mut Ctx,
    stream: &[Trav],
    body: &Traversal,
    times: Option<usize>,
    until: Option<&Traversal>,
    until_before: bool,
    emit: Option<&Traversal>,
    emit_before: bool,
) -> Vec<Trav> {
    const CAP: usize = 64;
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
        assert!(dedup_key(&GVal::List(vec![GVal::Num(1.0), GVal::Num(f64::NAN)])).is_none());
        assert!(dedup_key(&GVal::Map(vec![(
            GVal::Str(Arc::from("k")),
            GVal::Num(f64::NAN)
        )]))
        .is_none());
    }

    #[test]
    fn element_and_value_keys() {
        assert_eq!(dedup_key(&GVal::Vertex(7)), dedup_key(&GVal::Vertex(7)));
        assert_ne!(dedup_key(&GVal::Vertex(7)), dedup_key(&GVal::Vertex(8)));
        // same id, different element kind → different key (a vertex isn't an edge).
        assert_ne!(dedup_key(&GVal::Vertex(7)), dedup_key(&GVal::Edge(7)));
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
            dedup_key(&GVal::List(vec![GVal::Num(-0.0), GVal::Vertex(1)])),
            dedup_key(&GVal::List(vec![GVal::Num(0.0), GVal::Vertex(1)])),
        );
    }
}
