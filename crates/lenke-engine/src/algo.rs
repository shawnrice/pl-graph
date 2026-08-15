//! In-engine graph algorithms over [`crate::store::Store`]. Each returns a
//! per-node result keyed by the node's dense id, in ascending-id order, so the
//! output is DETERMINISTIC and reproducible (the same rule lenke-core follows so
//! the two engines agree). Deleted (tombstoned) nodes are skipped.
//!
//! The catalog: the non-iterative set — degree, weakly/strongly connected
//! components (union-by-min / Tarjan), on-cycle, BFS distances, single-source
//! shortest paths, closeness and betweenness centrality, and neighbor feature
//! aggregation — plus the iterative set, PageRank (global and personalized), label
//! propagation and peer-pressure clustering, whose f64 summation / tiebreak order
//! is pinned (node-id and
//! in-adjacency order, reciprocal multiply) so their results are reproducible and
//! match lenke-core bit-for-bit. Most procedures yield one scalar per node;
//! neighbor_aggregate yields a per-node feature vector (a `Value::List`).

use crate::ir::Dir;
use crate::store::Store;
use crate::value::Value;
use std::collections::HashMap;

/// PageRank defaults, matching lenke-core.
pub const DEFAULT_DAMPING: f64 = 0.85;
pub const DEFAULT_PAGERANK_ITERATIONS: u32 = 20;
/// Label-propagation default round bound, matching lenke-core.
pub const DEFAULT_LABEL_ITERATIONS: u32 = 10;
/// Peer-pressure default round bound, matching lenke-core (30, NOT the pagerank 20).
pub const DEFAULT_PEER_PRESSURE_ITERATIONS: u32 = 30;

/// Resolve an optional edge-type name to `Some(Some(id))` (a specific type),
/// `Some(None)` (any type), or `None` (a named-but-unknown type → matches
/// nothing). Kept separate so each algorithm handles the unknown-type case the
/// same way.
fn want_etype(store: &Store, edge_label: Option<&str>) -> Option<Option<u32>> {
    match edge_label {
        None => Some(None),
        Some(name) => store.etype_id(name).map(Some),
    }
}

/// Visit each `dir`/`want`-matching neighbour of `v`, calling `f(nbr)`. `want` is
/// `None` for any type or `Some(id)` for a specific one.
fn for_each_nbr(store: &Store, v: u32, dir: Dir, want: Option<u32>, mut f: impl FnMut(u32)) {
    let ok = |et: u32| want.is_none_or(|w| w == et);
    if matches!(dir, Dir::Out | Dir::Both) {
        for a in store.out(v) {
            if ok(a.etype) {
                f(a.nbr);
            }
        }
    }
    if matches!(dir, Dir::In | Dir::Both) {
        for a in store.inc(v) {
            if ok(a.etype) {
                f(a.nbr);
            }
        }
    }
}

/// A compact CSR snapshot of the adjacency for ONE `(dir, want)` — the flattened
/// neighbour ids `nbr` plus per-node offsets `off`. Built once from `for_each_nbr`
/// (so neighbour ORDER is preserved exactly — out-slice then in-slice, matching the
/// live traversal), it replaces the per-node `Vec<Vec<Adj>>` pointer-chase with a
/// contiguous 4-byte-per-edge sweep. Worth it only for MULTI-PASS algorithms
/// (PageRank, label propagation), where the O(E) build amortizes over many passes
/// and every pass then streams cache-friendly memory; the summation/visit order is
/// identical, so results stay byte-for-byte the same.
struct Csr {
    off: Vec<u32>,
    nbr: Vec<u32>,
}

impl Csr {
    fn build(store: &Store, dir: Dir, want: Option<u32>, slots: usize) -> Self {
        let mut off = vec![0u32; slots + 1];
        let mut nbr: Vec<u32> = Vec::new();
        for v in 0..slots as u32 {
            off[v as usize] = u32::try_from(nbr.len()).expect("edge count exceeds u32");
            for_each_nbr(store, v, dir, want, |u| nbr.push(u));
        }
        off[slots] = u32::try_from(nbr.len()).expect("edge count exceeds u32");
        Self { off, nbr }
    }

    #[inline]
    fn nbrs(&self, v: u32) -> &[u32] {
        &self.nbr[self.off[v as usize] as usize..self.off[v as usize + 1] as usize]
    }
}

/// Degree centrality: per live node, the count of incident edges along `dir`
/// (`Out`/`In`/`Both`), optionally restricted to `edge_label`. A named-but-unknown
/// edge type gives every node degree 0. Ascending-id order.
#[must_use]
pub fn degree(store: &Store, dir: Dir, edge_label: Option<&str>) -> Vec<(u32, f64)> {
    let nodes = store.all_nodes();
    let Some(want) = want_etype(store, edge_label) else {
        return nodes.into_iter().map(|v| (v, 0.0)).collect();
    };
    nodes
        .into_iter()
        .map(|v| {
            let mut d = 0u64;
            for_each_nbr(store, v, dir, want, |_| d += 1);
            (v, d as f64)
        })
        .collect()
}

/// Union-find root with full path compression.
fn find(parent: &mut [u32], x: u32) -> u32 {
    let mut root = x;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    let mut cur = x;
    while parent[cur as usize] != root {
        let next = parent[cur as usize];
        parent[cur as usize] = root;
        cur = next;
    }
    root
}

/// Union `a` and `b`, keeping the SMALLER-indexed root — so a component's id is
/// its lowest-dense-id member, independent of edge-processing order.
fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra == rb {
        return;
    }
    let (keep, drop) = if ra < rb { (ra, rb) } else { (rb, ra) };
    parent[drop as usize] = keep;
}

/// Weakly connected components: edges treated as UNDIRECTED, each node labelled by
/// its component's id = the smallest dense id in it (union-by-min, so the result
/// is independent of processing order). A named-but-unknown edge type → every node
/// its own component. Returns `(node, component_id)` in ascending-id order.
#[must_use]
pub fn weakly_connected_components(store: &Store, edge_label: Option<&str>) -> Vec<(u32, u32)> {
    let n = store.node_count();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    // A known type (or "any"): union each live node's out-edges (in-edges are their
    // mirrors, so out alone covers the undirected graph). Unknown type → no unions.
    if let Some(want) = want_etype(store, edge_label) {
        for &v in &store.all_nodes() {
            for_each_nbr(store, v, Dir::Out, want, |nbr| union(&mut parent, v, nbr));
        }
    }
    store
        .all_nodes()
        .into_iter()
        .map(|v| (v, find(&mut parent, v)))
        .collect()
}

/// Every SHORTEST path from `src` along `dir` to every reachable vertex, as node-id
/// lists (`[src, …, target]`). BFS records all minimum-length predecessors, then
/// reconstructs each distinct shortest path — a target reachable two equally-short
/// ways yields two paths. Ported from lenke-core's `shortest_paths_from`. The result
/// ORDER is unspecified (target set iterated in hash order), so compare as a multiset.
#[must_use]
pub fn shortest_paths_from(store: &Store, src: u32, dir: Dir) -> Vec<Vec<u32>> {
    use std::collections::HashMap;
    if (src as usize) >= store.node_count() || !store.is_alive(src) {
        return Vec::new();
    }
    let mut dist: HashMap<u32, usize> = HashMap::from([(src, 0)]);
    let mut preds: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut frontier = vec![src];
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for &v in &frontier {
            let d = dist[&v];
            let mut nbrs = Vec::new();
            for_each_nbr(store, v, dir, None, |n| nbrs.push(n));
            for n in nbrs {
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
    fn build(
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
            build(src, p, &path, preds, out);
        }
    }
    let mut paths = Vec::new();
    for &id in dist.keys() {
        build(src, id, &[], &preds, &mut paths);
    }
    paths
}

/// BFS hop distances from `source` along `dir`/`edge_label`. Returns `(node,
/// distance)` for every node REACHED (including the source at 0), in ascending-id
/// order. Distances are shortest-path, so the result is order-independent. A
/// named-but-unknown edge type reaches only the source.
#[must_use]
pub fn bfs_distances(
    store: &Store,
    source: u32,
    dir: Dir,
    edge_label: Option<&str>,
) -> Vec<(u32, u32)> {
    let n = store.node_count();
    let mut dist: Vec<Option<u32>> = vec![None; n];
    if (source as usize) >= n || !store.is_alive(source) {
        return Vec::new();
    }
    dist[source as usize] = Some(0);
    // Unknown edge type → only the source is reachable.
    if let Some(want) = want_etype(store, edge_label) {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source);
        while let Some(v) = queue.pop_front() {
            let d = dist[v as usize].expect("dequeued nodes are visited");
            for_each_nbr(store, v, dir, want, |nbr| {
                if dist[nbr as usize].is_none() {
                    dist[nbr as usize] = Some(d + 1);
                    queue.push_back(nbr);
                }
            });
        }
    }
    (0..n as u32)
        .filter_map(|v| dist[v as usize].map(|d| (v, d)))
        .collect()
}

/// PageRank (pull model): `pr'[v] = (1−d)/N + d·dangling/N + d·Σ_{u→v} pr[u]/outdeg[u]`,
/// where `dangling = Σ pr[u]` over out-degree-0 nodes and `N` = live node count.
/// Runs a FIXED `iterations` (no tolerance). Ported from lenke-core: the per-target
/// pull sums its terms in `in_adj` order (== edge-insertion order for that target),
/// and the dangling sum is in node-id order, so the accumulation is reproducible
/// (and matches lenke-core bit-for-bit on the same graph). Unweighted. Returns
/// `(node, rank)` in ascending-id order; a named-but-unknown edge type makes every
/// node dangling → the uniform `1/N`.
#[must_use]
pub fn pagerank(
    store: &Store,
    edge_label: Option<&str>,
    damping: f64,
    iterations: u32,
) -> Vec<(u32, f64)> {
    let live = store.all_nodes();
    let nf = live.len() as f64;
    if live.is_empty() {
        return Vec::new();
    }
    let Some(want) = want_etype(store, edge_label) else {
        return live.into_iter().map(|v| (v, 1.0 / nf)).collect();
    };
    let slots = store.node_count();

    // Out-degree per node (of the wanted type); a 0 marks a dangling node.
    let mut outdeg = vec![0u32; slots];
    for &u in &live {
        let mut c = 0u32;
        for_each_nbr(store, u, Dir::Out, want, |_| c += 1);
        outdeg[u as usize] = c;
    }

    let mut pr = vec![0.0f64; slots];
    for &v in &live {
        pr[v as usize] = 1.0 / nf;
    }

    // In-CSR built once: PageRank pulls over incoming edges every iteration, so the
    // contiguous sweep amortizes across all of them. Neighbour order equals
    // `for_each_nbr(Dir::In)` order, keeping the pinned summation order.
    let in_csr = Csr::build(store, Dir::In, want, slots);
    for _ in 0..iterations {
        // Dangling mass, summed in node-id order (serial — the f64 order is pinned).
        let mut dangling = 0.0;
        for &u in &live {
            if outdeg[u as usize] == 0 {
                dangling += pr[u as usize];
            }
        }
        let base = (1.0 - damping) / nf + damping * dangling / nf;
        let mut next = vec![0.0f64; slots];
        for &v in &live {
            // Pull over incoming edges u→v (in `in_adj` order): each source u has
            // this out-edge, so its out-degree is ≥ 1 (no divide-by-zero).
            let mut sum = 0.0;
            for &u in in_csr.nbrs(v) {
                sum += pr[u as usize] / f64::from(outdeg[u as usize]);
            }
            next[v as usize] = base + damping * sum;
        }
        pr = next;
    }
    live.into_iter().map(|v| (v, pr[v as usize])).collect()
}

/// Personalized PageRank (random-walk-with-restart): like PageRank, but the restart
/// mass teleports to a PERSONALIZATION vector `p` — uniform `1/k` over the `k`
/// distinct resolved `source_ext_ids` seeds (unknown/duplicate ids dropped), or
/// `1/N` when no seed resolves (degenerating to global PageRank). Ported from core's
/// `personalized_pagerank`: `teleport = (1-d) + d·dangling`; `pr'[v] = teleport·p[v]
/// + d·Σ_{u→v} pr[u]·recip[u]` with `recip[u] = 1/outdeg[u]` a PRECOMPUTED reciprocal
/// that is MULTIPLIED (matching core's `inc_fac`), the dangling sum taken in
/// ascending-id order and the incoming pull in in-adjacency (edge-insertion) order —
/// the same fixed orders core uses, so it is byte-identical. `pr` starts at `p`.
/// Returns `(node, score)` in ascending-id order.
#[must_use]
pub fn personalized_pagerank(
    store: &Store,
    edge_label: Option<&str>,
    source_ext_ids: &[String],
    damping: f64,
    iterations: u32,
) -> Vec<(u32, f64)> {
    let live = store.all_nodes();
    if live.is_empty() {
        return Vec::new();
    }
    let nf = live.len() as f64;
    let slots = store.node_count();
    let want_opt = want_etype(store, edge_label); // Option<Option<u32>>; None = no edges

    // Out-degree of the wanted type, and its precomputed reciprocal (multiplied, not
    // divided, to match core's `inc_fac` bit-for-bit). A dangling node's recip is 0.
    let mut outdeg = vec![0u32; slots];
    if let Some(want) = want_opt {
        for &u in &live {
            let mut c = 0u32;
            for_each_nbr(store, u, Dir::Out, want, |_| c += 1);
            outdeg[u as usize] = c;
        }
    }
    let recip: Vec<f64> = outdeg
        .iter()
        .map(|&d| if d > 0 { 1.0 / f64::from(d) } else { 0.0 })
        .collect();

    // Resolve seeds → dense slots, dedup keeping first, drop unknowns.
    let mut seed_slots: Vec<usize> = Vec::new();
    let mut seen = vec![false; slots];
    for id in source_ext_ids {
        if let Some(s) = store.node_by_ext(id) {
            let su = s as usize;
            if !seen[su] {
                seen[su] = true;
                seed_slots.push(su);
            }
        }
    }

    // Personalization vector: uniform over the seeds, or global `1/N` if none resolve.
    let mut p = vec![0.0f64; slots];
    if seed_slots.is_empty() {
        for &v in &live {
            p[v as usize] = 1.0 / nf;
        }
    } else {
        let share = 1.0 / seed_slots.len() as f64;
        for &s in &seed_slots {
            p[s] = share;
        }
    }
    let mut pr = p.clone();

    for _ in 0..iterations {
        // Dangling mass, summed in ascending-id order (pinned f64 order).
        let mut dangling = 0.0;
        for &u in &live {
            if outdeg[u as usize] == 0 {
                dangling += pr[u as usize];
            }
        }
        // Restart mass redistributed per `p` (not uniformly) — `teleport * p[v]`.
        let teleport = (1.0 - damping) + damping * dangling;
        let mut next = vec![0.0f64; slots];
        if let Some(want) = want_opt {
            for &v in &live {
                let mut sum = 0.0;
                for_each_nbr(store, v, Dir::In, want, |u| {
                    sum += pr[u as usize] * recip[u as usize];
                });
                next[v as usize] = teleport * p[v as usize] + damping * sum;
            }
        } else {
            // No edges: every pull sum is 0, so `pr` relaxes straight back to `p`.
            for &v in &live {
                next[v as usize] = teleport * p[v as usize];
            }
        }
        pr = next;
    }

    live.into_iter().map(|v| (v, pr[v as usize])).collect()
}

/// One node's winning label from the frozen `labels` snapshot: the most frequent label
/// among its `csr` neighbours, ties broken by smallest id. `None` = no neighbours (keep).
/// `count`/`touched` are reused scratch (a dense per-label tally reset via the touched
/// list, O(degree) per node); each parallel worker owns its own pair.
fn lp_pick(
    v: u32,
    csr: &Csr,
    labels: &[u32],
    count: &mut [u32],
    touched: &mut Vec<u32>,
) -> Option<u32> {
    touched.clear();
    for &u in csr.nbrs(v) {
        let lbl = labels[u as usize];
        if count[lbl as usize] == 0 {
            touched.push(lbl);
        }
        count[lbl as usize] += 1;
    }
    let mut best: Option<(u32, u32)> = None; // (label, count)
    for &lbl in touched.iter() {
        let c = count[lbl as usize];
        if best.is_none_or(|(bl, bc)| c > bc || (c == bc && lbl < bl)) {
            best = Some((lbl, c));
        }
    }
    for &lbl in touched.iter() {
        count[lbl as usize] = 0; // reset only what this node touched
    }
    best.map(|(lbl, _)| lbl)
}

/// One synchronous round: each node in `live` adopts its winning label from the frozen
/// `labels`. With `threads > 1` the work is split into scoped-thread chunks — every node's
/// result is independent, so the merged `next` is identical to the single-thread round
/// (same tally, same tie-break, order-independent). Each worker owns its dense tally.
fn lp_round(live: &[u32], csr: &Csr, labels: &[u32], n: usize, threads: usize) -> Vec<u32> {
    let mut next = labels.to_vec();
    if threads <= 1 {
        let mut count = vec![0u32; n];
        let mut touched: Vec<u32> = Vec::new();
        for &v in live {
            if let Some(lbl) = lp_pick(v, csr, labels, &mut count, &mut touched) {
                next[v as usize] = lbl;
            }
        }
        return next;
    }
    let chunk = live.len().div_ceil(threads);
    let parts: Vec<Vec<(u32, u32)>> = std::thread::scope(|s| {
        let handles: Vec<_> = live
            .chunks(chunk)
            .map(|ch| {
                s.spawn(move || {
                    let mut count = vec![0u32; n];
                    let mut touched: Vec<u32> = Vec::new();
                    let mut out = Vec::with_capacity(ch.len());
                    for &v in ch {
                        if let Some(lbl) = lp_pick(v, csr, labels, &mut count, &mut touched) {
                            out.push((v, lbl));
                        }
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("label-prop round"))
            .collect()
    });
    for part in &parts {
        for &(v, lbl) in part {
            next[v as usize] = lbl;
        }
    }
    next
}

/// Synchronous label propagation (community detection). Every node starts labelled
/// with its own id; each round it adopts the label most common among its UNDIRECTED
/// neighbours (out + in), ties broken by the SMALLEST label id. Rounds are
/// synchronous (read the frozen snapshot, commit together) for a fixed `iterations`
/// bound, stopping early once a round changes nothing. Returns `(node, label)` in
/// ascending-id order; a named-but-unknown edge type keeps every node its own label.
///
/// (lenke-core carries labels as external-id STRINGS and breaks ties
/// lexicographically; this engine has only dense node ids, so the tiebreak is the
/// smallest id — the results agree whenever id order matches string order.)
#[must_use]
pub fn label_propagation(
    store: &Store,
    edge_label: Option<&str>,
    iterations: u32,
) -> Vec<(u32, u32)> {
    let n = store.node_count();
    let live = store.all_nodes();
    let mut labels: Vec<u32> = (0..n as u32).collect();
    if let Some(want) = want_etype(store, edge_label) {
        // Both-direction CSR built once and swept every iteration.
        let csr = Csr::build(store, Dir::Both, want, n);
        // A round is embarrassingly parallel — every node reads the frozen `labels` and
        // its result is independent — so split `live` across scoped threads (the engine
        // has no rayon; std::thread::scope is its parallel primitive, as the traversal
        // path uses). Each worker owns its dense tally and returns (node, label) pairs
        // that the main thread applies, so the result is byte-identical to a sequential
        // round. Small graphs stay single-threaded (thread setup would dominate).
        let threads = if live.len() >= 8192 {
            std::thread::available_parallelism().map_or(1, std::num::NonZero::get).min(8)
        } else {
            1
        };
        for _ in 0..iterations {
            let next = lp_round(&live, &csr, &labels, n, threads);
            if next == labels {
                break; // converged
            }
            labels = next;
        }
    }
    live.into_iter().map(|v| (v, labels[v as usize])).collect()
}

/// Peer-pressure clustering (a directed, vote-weighted label propagation). Every
/// vertex starts as its own cluster; each round it adopts the cluster with the
/// highest incoming VOTE ENERGY, where a source `s` casts `vote[s] = 1/outdeg[s]`
/// for its current cluster and the energies are summed over `s`'s in-neighbours in
/// in-adjacency (edge-insertion) order — the same fixed order core uses, so the
/// per-cluster f64 sum is byte-identical. A vertex with no incoming vote keeps its
/// own cluster. Ties are broken by the SMALLEST external-id string (matching core's
/// `vid.text` comparison, not the dense id, so multi-digit ids agree too). Rounds
/// run to convergence or `iterations` (core's default is 30). The cluster label is
/// the winning cluster's dense id (surfaced as a number, like WCC/SCC). Returns
/// `(node, cluster)` in ascending-id order; a named-but-unknown edge type → every
/// vertex its own cluster.
#[must_use]
pub fn peer_pressure(store: &Store, edge_label: Option<&str>, iterations: u32) -> Vec<(u32, u32)> {
    let n = store.node_count();
    let live = store.all_nodes();
    let mut cluster: Vec<u32> = (0..n as u32).collect();
    let Some(want) = want_etype(store, edge_label) else {
        return live.into_iter().map(|v| (v, v)).collect();
    };

    // vote[u] = 1/out-degree[u] (of the wanted type); a 0-out vertex casts nothing.
    let mut outdeg = vec![0u32; n];
    for &u in &live {
        let mut c = 0u32;
        for_each_nbr(store, u, Dir::Out, want, |_| c += 1);
        outdeg[u as usize] = c;
    }
    let vote: Vec<f64> = outdeg
        .iter()
        .map(|&d| if d > 0 { 1.0 / f64::from(d) } else { 0.0 })
        .collect();
    // Break ties on the source cluster's EXTERNAL id string (core's `vid.text`).
    let ext = |c: u32| store.node_ext_id(c);

    for _ in 0..iterations {
        let mut next = cluster.clone();
        for &u in &live {
            // Tally incoming vote energy per candidate cluster, in in-adjacency order.
            let mut energy: HashMap<u32, f64> = HashMap::new();
            let mut any = false;
            for a in store.inc(u) {
                if want.is_none_or(|w| w == a.etype) {
                    *energy.entry(cluster[a.nbr as usize]).or_insert(0.0) += vote[a.nbr as usize];
                    any = true;
                }
            }
            if !any {
                continue; // no incoming vote → keep own cluster
            }
            // Adopt the max-energy cluster; tie → smallest external id.
            let mut best: Option<(u32, f64)> = None;
            for (&c, &e) in &energy {
                let better = match best {
                    None => true,
                    Some((bc, be)) => e > be || (e == be && ext(c) < ext(bc)),
                };
                if better {
                    best = Some((c, e));
                }
            }
            if let Some((c, _)) = best {
                next[u as usize] = c;
            }
        }
        if next == cluster {
            break; // converged
        }
        cluster = next;
    }

    live.into_iter().map(|v| (v, cluster[v as usize])).collect()
}

/// Closeness centrality (unweighted, directed OUT): for each node, the reciprocal
/// of the summed shortest-path (hop) distances to every node it can reach, or 0
/// when it reaches nothing else. Ported from lenke-core's `closeness`: the
/// distances are integer BFS hops summed in ascending-id order and the only
/// floating-point operation is the final reciprocal, so it is byte-identical to
/// core on the same graph. Weighted closeness (`weightProperty`) is deferred — this
/// is the unweighted default. Returns `(node, closeness)` in ascending-id order; a
/// named-but-unknown edge type reaches only each source (every sum 0 → every 0).
#[must_use]
pub fn closeness(
    store: &Store,
    edge_label: Option<&str>,
    weight_property: Option<&str>,
) -> Vec<(u32, f64)> {
    store
        .all_nodes()
        .into_iter()
        .map(|s| {
            // Sum finite shortest-path distances in ascending-id order (core's
            // vertex-insertion order). Unweighted → BFS hops; weighted → Dijkstra.
            let sum = match weight_property {
                None => {
                    // `bfs_distances` yields reached nodes (incl. the source at 0),
                    // ascending id — the same set/order core sums.
                    let mut acc = 0.0f64;
                    for (_, d) in bfs_distances(store, s, Dir::Out, edge_label) {
                        acc += f64::from(d);
                    }
                    acc
                }
                Some(wk) => {
                    let dist = dijkstra_dist(store, s, Dir::Out, edge_label, wk);
                    let mut acc = 0.0f64;
                    for d in dist {
                        if d.is_finite() {
                            acc += d;
                        }
                    }
                    acc
                }
            };
            let c = if sum == 0.0 { 0.0 } else { 1.0 / sum };
            (s, c)
        })
        .collect()
}

/// Strongly connected components (Tarjan, iterative): partition the directed graph
/// into maximal sets of mutually reachable vertices, labelling every member with
/// the SMALLEST dense id in its component (matching the WCC convention). Ported
/// from lenke-core's `strongly_connected_components`, which uses the same iterative
/// Tarjan and the same min-member representative — the partition is unique and the
/// rep is order-independent, so this agrees with core regardless of adjacency
/// order. (Core surfaces the rep as its external-id string; this engine has only
/// dense ids, so the rep is the number, exactly as WCC's `connected_components`
/// does.) Returns `(node, representative)` in ascending-id order; a named-but-
/// unknown edge type leaves every vertex a singleton (its own rep).
#[must_use]
pub fn strongly_connected_components(store: &Store, edge_label: Option<&str>) -> Vec<(u32, u32)> {
    let n = store.node_count();
    // Forward out-adjacency of the wanted type as a FLAT CSR (offsets + neighbours) —
    // ONE allocation, not a `Vec` per vertex (a million small allocations at 1M
    // vertices dominated the runtime). Neighbour order within a vertex is the
    // `for_each_nbr` order, unchanged, so the DFS visits identically; unknown type →
    // no edges. `adj[v]` becomes `flat[offsets[v]..offsets[v+1]]`.
    let (offsets, flat): (Vec<u32>, Vec<u32>) = match want_etype(store, edge_label) {
        None => (vec![0u32; n + 1], Vec::new()),
        Some(want) => {
            let mut offsets = vec![0u32; n + 1];
            for &v in &store.all_nodes() {
                for_each_nbr(store, v, Dir::Out, want, |_| offsets[v as usize + 1] += 1);
            }
            for v in 0..n {
                offsets[v + 1] += offsets[v];
            }
            let mut flat = vec![0u32; offsets[n] as usize];
            let mut cur = offsets.clone();
            for &v in &store.all_nodes() {
                for_each_nbr(store, v, Dir::Out, want, |nbr| {
                    flat[cur[v as usize] as usize] = nbr;
                    cur[v as usize] += 1;
                });
            }
            (offsets, flat)
        }
    };
    let adj_of = |v: usize| -> &[u32] { &flat[offsets[v] as usize..offsets[v + 1] as usize] };

    const UNVISITED: u32 = u32::MAX;
    let mut order = vec![UNVISITED; n]; // DFS discovery index (Tarjan's `index`)
    let mut low = vec![0u32; n]; // lowlink
    let mut on_stack = vec![false; n];
    let mut comp = vec![UNVISITED; n]; // resolved component representative
    let mut tstack: Vec<u32> = Vec::new(); // Tarjan's component stack
    let mut counter: u32 = 0;
    // Each DFS frame is `(vertex, next-neighbour cursor into adj[v])`.
    let mut frames: Vec<(u32, usize)> = Vec::new();

    for &s in &store.all_nodes() {
        if order[s as usize] != UNVISITED {
            continue;
        }
        order[s as usize] = counter;
        low[s as usize] = counter;
        counter += 1;
        on_stack[s as usize] = true;
        tstack.push(s);
        frames.push((s, 0));

        while let Some(&(v, ci)) = frames.last() {
            let vu = v as usize;
            let nbrs = adj_of(vu);
            if ci < nbrs.len() {
                frames.last_mut().unwrap().1 = ci + 1;
                let w = nbrs[ci];
                let wu = w as usize;
                if order[wu] == UNVISITED {
                    order[wu] = counter;
                    low[wu] = counter;
                    counter += 1;
                    on_stack[wu] = true;
                    tstack.push(w);
                    frames.push((w, 0));
                } else if on_stack[wu] {
                    low[vu] = low[vu].min(order[wu]);
                }
            } else {
                // `v` fully explored: an SCC root pops its whole component and stamps
                // every member with the component's smallest dense id.
                if low[vu] == order[vu] {
                    let mut members: Vec<u32> = Vec::new();
                    loop {
                        let m = tstack.pop().expect("component stack non-empty at root");
                        on_stack[m as usize] = false;
                        members.push(m);
                        if m == v {
                            break;
                        }
                    }
                    let rep = *members.iter().min().expect("a component has a member");
                    for m in members {
                        comp[m as usize] = rep;
                    }
                }
                frames.pop();
                if let Some(&(p, _)) = frames.last() {
                    low[p as usize] = low[p as usize].min(low[vu]);
                }
            }
        }
    }

    store
        .all_nodes()
        .into_iter()
        .map(|v| (v, comp[v as usize]))
        .collect()
}

/// Per-vertex cycle membership: `1.0` iff the vertex lies on a directed cycle —
/// its SCC has more than one member OR it has a self-loop (a 1-cycle) — else `0.0`.
/// Ported from core's `on_cycle`, derived from the same SCC partition plus a
/// self-loop scan, so it is byte-identical (the value is a boolean 0/1, no float
/// arithmetic). Returns `(node, on_cycle)` in ascending-id order; a named-but-
/// unknown edge type → no edges → every vertex `0.0`.
#[must_use]
pub fn on_cycle(store: &Store, edge_label: Option<&str>) -> Vec<(u32, f64)> {
    let n = store.node_count();
    // Component sizes by representative (a component with >1 member is a cycle).
    let comp = strongly_connected_components(store, edge_label);
    let mut rep_of = vec![0u32; n];
    let mut size = vec![0u32; n];
    for &(v, r) in &comp {
        rep_of[v as usize] = r;
        size[r as usize] += 1;
    }
    // Self-loops (v→v of the wanted type) put a singleton on a 1-cycle too.
    let mut self_loop = vec![false; n];
    if let Some(want) = want_etype(store, edge_label) {
        for &v in &store.all_nodes() {
            for_each_nbr(store, v, Dir::Out, want, |nbr| {
                if nbr == v {
                    self_loop[v as usize] = true;
                }
            });
        }
    }
    store
        .all_nodes()
        .into_iter()
        .map(|v| {
            let cyclic = size[rep_of[v as usize] as usize] > 1 || self_loop[v as usize];
            (v, if cyclic { 1.0 } else { 0.0 })
        })
        .collect()
}

/// Betweenness centrality (Brandes, unweighted, directed OUT, exact all-sources):
/// for each vertex, the sum over all source-target pairs of the fraction of
/// shortest paths that pass through it. Ported from core's `betweenness`: the
/// per-source SSSP is the same BFS (VecDeque, `stack` in dequeue order, neighbours
/// in edge-insertion order — which the engine's adjacency already is), `sigma`
/// counts shortest paths, `pred` the predecessors, and the dependency
/// back-propagation runs in reverse-stack order — the same fixed order core uses,
/// so the per-vertex f64 sum is byte-identical. Sources are taken in ascending id.
/// Weighted betweenness (`weightProperty`) and pivot sampling (`pivots`) are
/// deferred — this is the exact unweighted default. Returns `(node, centrality)` in
/// ascending-id order; a named-but-unknown edge type → every vertex `0.0`.
#[must_use]
pub fn betweenness(
    store: &Store,
    edge_label: Option<&str>,
    weight_property: Option<&str>,
) -> Vec<(u32, f64)> {
    let n = store.node_count();
    let live = store.all_nodes();
    let want = want_etype(store, edge_label);
    let type_ok = |et: u32| want.is_some_and(|inner| inner.is_none_or(|t| t == et));
    let mut cb = vec![0f64; n];

    for &s in &live {
        // --- single-source shortest-path DAG (sigma / pred / stack) ---
        let mut sigma = vec![0f64; n];
        let mut pred: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut dist = vec![f64::INFINITY; n];
        let mut stack: Vec<u32> = Vec::new();
        sigma[s as usize] = 1.0;
        dist[s as usize] = 0.0;
        if let Some(wk) = weight_property {
            // Weighted Dijkstra SSSP: settle in (dist, idx) heap order (== stack
            // order); sigma RESETS on a strictly shorter path, ACCUMULATES on an
            // equal-length one — core's weighted `sssp` branch exactly.
            let mut settled = vec![false; n];
            let mut heap = std::collections::BinaryHeap::new();
            heap.push(DijkstraState { dist: 0.0, idx: s });
            while let Some(DijkstraState { idx: v, .. }) = heap.pop() {
                if settled[v as usize] {
                    continue;
                }
                settled[v as usize] = true;
                stack.push(v);
                let dv = dist[v as usize];
                for a in store.out(v) {
                    if !type_ok(a.etype) {
                        continue;
                    }
                    let to = a.nbr;
                    let nd = dv + edge_weight(store, a.eid, wk);
                    if nd < dist[to as usize] {
                        dist[to as usize] = nd;
                        sigma[to as usize] = sigma[v as usize];
                        pred[to as usize] = vec![v];
                        heap.push(DijkstraState { dist: nd, idx: to });
                    } else if nd == dist[to as usize] {
                        sigma[to as usize] += sigma[v as usize];
                        pred[to as usize].push(v);
                    }
                }
            }
        } else {
            // Unweighted BFS layers.
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(s);
            while let Some(v) = queue.pop_front() {
                stack.push(v);
                let dv = dist[v as usize];
                if let Some(w) = want {
                    for_each_nbr(store, v, Dir::Out, w, |to| {
                        if dist[to as usize].is_infinite() {
                            dist[to as usize] = dv + 1.0;
                            queue.push_back(to);
                        }
                        // A shortest-path edge (v is one hop closer than `to`): count
                        // v's paths into `to` and record v as a predecessor.
                        if dist[to as usize] == dv + 1.0 {
                            sigma[to as usize] += sigma[v as usize];
                            pred[to as usize].push(v);
                        }
                    });
                }
            }
        }

        // --- dependency accumulation, reverse-stack (non-increasing distance) ---
        let mut delta = vec![0f64; n];
        for &w in stack.iter().rev() {
            let coeff = 1.0 + delta[w as usize];
            for &v in &pred[w as usize] {
                delta[v as usize] += (sigma[v as usize] / sigma[w as usize]) * coeff;
            }
            if w != s {
                cb[w as usize] += delta[w as usize];
            }
        }
    }

    live.into_iter().map(|v| (v, cb[v as usize])).collect()
}

/// Single-source shortest-path distances (unweighted BFS layers) from a `source`
/// external id, along `dir`/`edge_label`. Returns `(node, distance)` for every node
/// REACHED (the source at 0), in ascending-id order — the unweighted default of
/// core's `shortestPath`. Distances are integer hops, so the result is byte-
/// identical to core. An unknown/absent source (or one resolving to no live node)
/// yields nothing; a named-but-unknown edge type reaches only the source. Weighted
/// (`weightProperty`, Dijkstra) and A* (`target`) are deferred.
#[must_use]
pub fn shortest_path(
    store: &Store,
    source: Option<&str>,
    dir: Dir,
    edge_label: Option<&str>,
    weight_property: Option<&str>,
    target: Option<&str>,
) -> Vec<(u32, f64)> {
    let Some(src) = source.and_then(|s| store.node_by_ext(s)) else {
        return Vec::new();
    };
    let all: Vec<(u32, f64)> = match weight_property {
        None => bfs_distances(store, src, dir, edge_label)
            .into_iter()
            .map(|(v, d)| (v, f64::from(d)))
            .collect(),
        Some(wk) => dijkstra(store, src, dir, edge_label, wk),
    };
    // A `target` restricts the result to that one vertex's distance (like core): its
    // row if reachable (`all` holds only reachable nodes), else nothing — and an
    // unknown target id yields nothing.
    match target {
        None => all,
        Some(t) => match store.node_by_ext(t) {
            Some(tgt) => all.into_iter().filter(|&(v, _)| v == tgt).collect(),
            None => Vec::new(),
        },
    }
}

/// A Dijkstra frontier entry, ordered as a MIN-heap on `(dist, idx)` — the exact
/// reversed comparison lenke-core's `State` uses, so `BinaryHeap` pops the smallest
/// distance (ties by smallest id) in the same order and the per-vertex f64 distance
/// is byte-identical.
struct DijkstraState {
    dist: f64,
    idx: u32,
}
impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.idx == other.idx
    }
}
impl Eq for DijkstraState {}
impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}
impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// An edge's weight for the weighted shortest path: its `key` property as `f64`, or
/// `0.0` when absent/non-numeric — matching core's `edge_weights`.
fn edge_weight(store: &Store, eid: u32, key: &str) -> f64 {
    match store.edge_prop(eid, key) {
        Value::Num(n) => n,
        _ => 0.0,
    }
}

/// Weighted single-source shortest paths (Dijkstra) from `src` along `dir`/
/// `edge_label`, edge costs read from the `weight` property. Ported from core's
/// weighted `shortestPath`: the same min-heap `(dist, idx)` ordering, stale-entry
/// skip (`du > dist[u]`), and strict relaxation (`nd < dist[nbr]`), with edge
/// weights read in the same way — so the settling order and the accumulated f64
/// distances are byte-identical. Core rejects a negative or NaN weight (Dijkstra is
/// undefined for them); here that yields the empty result. Returns `(node,
/// distance)` for every reachable node, ascending id.
fn dijkstra(
    store: &Store,
    src: u32,
    dir: Dir,
    edge_label: Option<&str>,
    weight: &str,
) -> Vec<(u32, f64)> {
    // Negative/NaN weights make Dijkstra unsound — reject the whole run (core errs
    // here, only for shortestPath; closeness/betweenness run the SSSP unchecked).
    for eid in store.all_edges() {
        let w = edge_weight(store, eid, weight);
        if w.is_nan() || w < 0.0 {
            return Vec::new();
        }
    }
    let dist = dijkstra_dist(store, src, dir, edge_label, weight);
    (0..dist.len() as u32)
        .filter(|&v| dist[v as usize].is_finite())
        .map(|v| (v, dist[v as usize]))
        .collect()
}

/// The weighted single-source distance vector (`f64::INFINITY` for unreachable) —
/// core's weighted `sssp`. Shared by weighted `shortest_path` and `closeness`. No
/// weight validation (the caller decides); relaxes incident edges in the configured
/// direction in adjacency order, so the settled distances are byte-identical to core.
fn dijkstra_dist(
    store: &Store,
    src: u32,
    dir: Dir,
    edge_label: Option<&str>,
    weight: &str,
) -> Vec<f64> {
    let n = store.node_count();
    let want = want_etype(store, edge_label);
    let type_ok = |et: u32| want.is_some_and(|inner| inner.is_none_or(|t| t == et));

    let mut dist = vec![f64::INFINITY; n];
    dist[src as usize] = 0.0;
    let mut heap = std::collections::BinaryHeap::new();
    heap.push(DijkstraState {
        dist: 0.0,
        idx: src,
    });
    while let Some(DijkstraState { dist: du, idx: u }) = heap.pop() {
        if du > dist[u as usize] {
            continue; // a shorter distance was already settled
        }
        // Relax the incident edges in the configured direction (out-adj then in-adj
        // for `both`), in adjacency (edge-insertion) order — core's visit_adj order.
        let mut relax = |nbr: u32, eid: u32, etype: u32| {
            if !type_ok(etype) {
                return;
            }
            let nd = du + edge_weight(store, eid, weight);
            if nd < dist[nbr as usize] {
                dist[nbr as usize] = nd;
                heap.push(DijkstraState { dist: nd, idx: nbr });
            }
        };
        if matches!(dir, Dir::Out | Dir::Both) {
            for a in store.out(u) {
                relax(a.nbr, a.eid, a.etype);
            }
        }
        if matches!(dir, Dir::In | Dir::Both) {
            for a in store.inc(u) {
                relax(a.nbr, a.eid, a.etype);
            }
        }
    }
    dist
}

/// A\* goal-directed search from `source` to `target` (both external ids). Like
/// Dijkstra, but the heap priority is `g + h`, where `g` is the best-known cost so
/// far and `h(v)` is the admissible lower-bound from `v` to the target read from the
/// `heuristic` property (0 when absent). Ported from core's `astar`: same
/// `(priority, idx)` heap order, same closed-set + strict `ng < g[nbr]` relaxation,
/// same edge weights (unweighted → unit cost). With an admissible heuristic the
/// settled `g[target]` is the exact shortest distance — identical to Dijkstra's, and
/// byte-identical to core. Returns `[(target, distance)]` if reachable, else empty.
fn astar_search(
    store: &Store,
    source: Option<&str>,
    target: Option<&str>,
    dir: Dir,
    edge_label: Option<&str>,
    weight: Option<&str>,
    heuristic: Option<&str>,
) -> Vec<(u32, f64)> {
    let (Some(src), Some(tgt)) = (
        source.and_then(|s| store.node_by_ext(s)),
        target.and_then(|t| store.node_by_ext(t)),
    ) else {
        return Vec::new();
    };
    let n = store.node_count();
    let want = want_etype(store, edge_label);
    let type_ok = |et: u32| want.is_some_and(|inner| inner.is_none_or(|t| t == et));
    let h = |v: u32| -> f64 {
        heuristic.map_or(0.0, |k| match store.prop(v, k) {
            Value::Num(x) => x,
            _ => 0.0,
        })
    };
    let cost = |eid: u32| weight.map_or(1.0, |k| edge_weight(store, eid, k));

    let mut g = vec![f64::INFINITY; n];
    g[src as usize] = 0.0;
    let mut closed = vec![false; n];
    let mut heap = std::collections::BinaryHeap::new();
    heap.push(DijkstraState {
        dist: h(src),
        idx: src,
    });
    while let Some(DijkstraState { idx: u, .. }) = heap.pop() {
        if closed[u as usize] {
            continue;
        }
        closed[u as usize] = true;
        if u == tgt {
            return vec![(tgt, g[u as usize])];
        }
        let mut relax = |nbr: u32, eid: u32, etype: u32| {
            if !type_ok(etype) || closed[nbr as usize] {
                return;
            }
            let ng = g[u as usize] + cost(eid);
            if ng < g[nbr as usize] {
                g[nbr as usize] = ng;
                heap.push(DijkstraState {
                    dist: ng + h(nbr),
                    idx: nbr,
                });
            }
        };
        if matches!(dir, Dir::Out | Dir::Both) {
            for a in store.out(u) {
                relax(a.nbr, a.eid, a.etype);
            }
        }
        if matches!(dir, Dir::In | Dir::Both) {
            for a in store.inc(u) {
                relax(a.nbr, a.eid, a.etype);
            }
        }
    }
    Vec::new()
}

/// The element-wise aggregation for [`neighbor_aggregate`].
#[derive(Clone, Copy, PartialEq)]
enum AggOp {
    Sum,
    Mean,
    Max,
    Min,
}

/// A vertex's feature vector: a numeric list-valued property `key`, as `Vec<f64>`.
/// `None` if absent or holding a non-numeric element (not a feature vector).
fn read_feature(store: &Store, v: u32, key: &str) -> Option<Vec<f64>> {
    match store.prop(v, key) {
        Value::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in &items {
                match it {
                    Value::Num(n) => out.push(*n),
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// Neighbor feature aggregation (a GNN message-passing primitive): for each node,
/// element-wise aggregate the `feature` vector property over its neighbours under
/// `op` (mean/sum/max/min). Ported from core's `neighbor_aggregate` for the
/// unweighted, un-normalized case: contributors are the node's out- and/or in-
/// neighbours (per `direction`, a both-direction self-loop counted once), gathered
/// and SORTED BY EDGE ID for a canonical accumulation order, so the folded f64 sum
/// is byte-identical to core. `mean` divides by the number of folded contributors;
/// a node with no featured contributor yields the zero vector (of the shared
/// dimension). `include_self` folds the node's own feature first. Weighted
/// (`weightProperty`) and GCN normalization are deferred (rejected for max/min in
/// core; not built here). Returns `(node, Value::List)` in ascending-id order, or an
/// `Err` for a missing `feature`, a bad `op`/`direction`, or ragged feature lengths.
fn neighbor_aggregate(
    store: &Store,
    config: &[(String, Value)],
) -> Result<Vec<(u32, Value)>, String> {
    let cfg_str = |k: &str| -> Option<&str> {
        config.iter().find(|(ck, _)| ck == k).and_then(|(_, v)| {
            if let Value::Str(s) = v {
                Some(s.as_ref())
            } else {
                None
            }
        })
    };
    let cfg_bool = |k: &str| -> Option<bool> {
        config.iter().find(|(ck, _)| ck == k).and_then(|(_, v)| {
            if let Value::Bool(b) = v {
                Some(*b)
            } else {
                None
            }
        })
    };
    let feature = cfg_str("feature")
        .ok_or_else(|| "neighbor_aggregate requires a `feature` property".to_string())?;
    let op = match cfg_str("op").unwrap_or("mean") {
        "mean" => AggOp::Mean,
        "sum" => AggOp::Sum,
        "max" => AggOp::Max,
        "min" => AggOp::Min,
        other => {
            return Err(format!(
                "neighbor_aggregate `op` must be one of mean|sum|max|min, got '{other}'"
            ))
        }
    };
    let (want_out, want_in) = match cfg_str("direction").unwrap_or("both") {
        "out" => (true, false),
        "in" => (false, true),
        "both" => (true, true),
        other => {
            return Err(format!(
                "neighbor_aggregate `direction` must be one of out|in|both, got '{other}'"
            ))
        }
    };
    let include_self = cfg_bool("includeSelf").unwrap_or(false);
    let want = want_etype(store, cfg_str("edgeType"));
    let gcn = match cfg_str("norm").unwrap_or("none") {
        "none" => false,
        "gcn" => true,
        other => {
            return Err(format!(
                "neighbor_aggregate `norm` must be one of none|gcn, got '{other}'"
            ))
        }
    };
    // A per-edge weight or a GCN norm SCALES each contributor, which is meaningless
    // for the order/scale-independent max/min — reject loudly (matching core).
    let weight_property = cfg_str("weightProperty");
    if (weight_property.is_some() || gcn) && matches!(op, AggOp::Max | AggOp::Min) {
        return Err(
            "neighbor_aggregate `weightProperty`/`norm` apply only to op=sum|mean (max/min are \
             scale-independent)"
                .to_string(),
        );
    }

    // Precompute every vertex's feature vector; infer the shared dimension (ragged
    // vectors fault, matching core).
    let slots = store.node_count();
    let feats: Vec<Option<Vec<f64>>> = (0..slots as u32)
        .map(|v| read_feature(store, v, feature))
        .collect();
    let mut dim: Option<usize> = None;
    for f in feats.iter().flatten() {
        match dim {
            None => dim = Some(f.len()),
            Some(d) if d != f.len() => {
                return Err(format!(
                "neighbor_aggregate feature vectors must all have the same length; found {} and {}",
                d,
                f.len()
            ))
            }
            _ => {}
        }
    }
    let d = dim.unwrap_or(0);

    // A vertex's contributor `(eid, nbr)` pairs, sorted by edge id — the canonical,
    // engine-independent accumulation order. A both-direction self-loop is counted
    // once (its in-side copy dropped, mirroring `expand`).
    let contributors = |v: u32| -> Vec<(u32, u32)> {
        let mut contrib: Vec<(u32, u32)> = Vec::new();
        if want_out {
            for a in store.out(v) {
                if want.is_some_and(|w| w.is_none_or(|t| t == a.etype)) {
                    contrib.push((a.eid, a.nbr));
                }
            }
        }
        if want_in {
            for a in store.inc(v) {
                if want_out && a.nbr == v {
                    continue;
                }
                if want.is_some_and(|w| w.is_none_or(|t| t == a.etype)) {
                    contrib.push((a.eid, a.nbr));
                }
            }
        }
        contrib.sort_unstable_by_key(|&(eid, _)| eid);
        contrib
    };

    // GCN degree per vertex: its contributor count under the same direction/type
    // filter, plus the self-loop when includeSelf (the Ã = A + I self-loop), floored
    // at 1 so a source/sink contributes a finite 1/sqrt(deg_i·deg_j) rather than
    // dividing by zero. Only needed for the GCN norm.
    let deg: Vec<f64> = if gcn {
        (0..slots as u32)
            .map(|v| (contributors(v).len() + usize::from(include_self)).max(1) as f64)
            .collect()
    } else {
        Vec::new()
    };

    let mut out: Vec<(u32, Value)> = Vec::with_capacity(store.all_nodes().len());
    for v in store.all_nodes() {
        let mut acc = vec![0.0f64; d];
        let mut coef_sum = 0.0f64; // Σ contributor coefficients (the `mean` divisor)
        let mut started = false; // whether `acc` holds a real value (for max/min)
        let mut fold = |vec: &[f64], coef: f64| {
            match op {
                AggOp::Sum | AggOp::Mean => {
                    for (a, x) in acc.iter_mut().zip(vec) {
                        *a += coef * *x;
                    }
                }
                AggOp::Max => {
                    if started {
                        for (a, x) in acc.iter_mut().zip(vec) {
                            *a = a.max(*x);
                        }
                    } else {
                        acc.copy_from_slice(vec);
                    }
                }
                AggOp::Min => {
                    if started {
                        for (a, x) in acc.iter_mut().zip(vec) {
                            *a = a.min(*x);
                        }
                    } else {
                        acc.copy_from_slice(vec);
                    }
                }
            }
            started = true;
            coef_sum += coef;
        };
        if include_self {
            if let Some(sv) = &feats[v as usize] {
                // The self-loop has weight 1.0 and GCN factor 1/deg_i (== 1/sqrt(deg·deg)).
                let coef = if gcn { 1.0 / deg[v as usize] } else { 1.0 };
                fold(sv, coef);
            }
        }
        for (eid, nbr) in contributors(v) {
            if let Some(nv) = &feats[nbr as usize] {
                // coef = edge weight (1.0 unweighted) × GCN factor (1/sqrt(deg_i·deg_j)).
                let w = weight_property.map_or(1.0, |wk| edge_weight(store, eid, wk));
                let nf = if gcn {
                    1.0 / (deg[v as usize] * deg[nbr as usize]).sqrt()
                } else {
                    1.0
                };
                fold(nv, w * nf);
            }
        }
        if op == AggOp::Mean && coef_sum != 0.0 {
            for a in &mut acc {
                *a /= coef_sum;
            }
        }
        out.push((v, Value::List(acc.into_iter().map(Value::Num).collect())));
    }
    Ok(out)
}

/// The built-in procedure catalog: a `CALL name(...)` procedure name → its
/// non-`node` result column name (matching lenke-core's snake_case surface). The
/// output columns of every procedure are `[node, <result>]`. `None` = unknown.
#[must_use]
pub fn procedure_result_col(name: &str) -> Option<&'static str> {
    Some(match name {
        "degree" => "degree",
        "pagerank" => "score",
        "connected_components" => "componentId",
        "strongly_connected_components" => "componentId",
        "on_cycle" => "onCycle",
        "label_propagation" => "label",
        "closeness" => "centrality",
        "betweenness" => "centrality",
        "shortest_path" => "distance",
        "personalized_pagerank" => "score",
        "neighbor_aggregate" => "vector",
        "peer_pressure" => "cluster",
        _ => return None,
    })
}

/// The accepted `CALL <algo>({…})` config keys (core's list, plus the engine's
/// `edgeType` spelling of `edgeLabel`). A config map is validated against this set so
/// a typo or a wrong key no longer silently no-ops. Order is fixed so the "did you
/// mean" tie-break is deterministic.
const CONFIG_KEYS: &[&str] = &[
    "edgeLabel",
    "edgeType",
    "direction",
    "weightProperty",
    "dampingFactor",
    "iterations",
    "pivots",
    "seedProperty",
    "source",
    "sourceNodes",
    "target",
    "writeProperty",
    "algorithm",
    "heuristicProperty",
    "feature",
    "op",
    "includeSelf",
    "norm",
];

/// Case-insensitive Levenshtein edit distance — for the config-key "did you mean".
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().flat_map(char::to_lowercase).collect();
    let b: Vec<char> = b.chars().flat_map(char::to_lowercase).collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Validate a `CALL` config map: every key must be a known config key (unknown →
/// error, with a "did you mean") and carry the right value TYPE (a wrong type is a
/// data exception). Matches core — a bad config no longer silently no-ops.
pub fn validate_config(config: &[(String, Value)]) -> Result<(), String> {
    for (k, v) in config {
        let expect: &str = match k.as_str() {
            "dampingFactor" | "iterations" | "pivots" => "number",
            "includeSelf" => "boolean",
            "sourceNodes" => "list",
            "edgeLabel" | "edgeType" | "direction" | "weightProperty" | "seedProperty"
            | "source" | "target" | "writeProperty" | "algorithm" | "heuristicProperty"
            | "feature" | "op" | "norm" => "string",
            _ => {
                let hint = CONFIG_KEYS
                    .iter()
                    .map(|c| (edit_distance(k, c), *c))
                    .filter(|(d, _)| *d <= 2)
                    .min_by_key(|(d, _)| *d)
                    .map(|(_, c)| c);
                return Err(match hint {
                    Some(s) => format!("unknown config key '{k}' (did you mean '{s}'?)"),
                    None => format!("unknown config key '{k}'"),
                });
            }
        };
        let ok = match expect {
            "number" => matches!(v, Value::Num(_)),
            "boolean" => matches!(v, Value::Bool(_)),
            "list" => matches!(v, Value::List(_)),
            _ => matches!(v, Value::Str(_)),
        };
        if !ok {
            return Err(format!("config key '{k}' expects a {expect}"));
        }
    }
    Ok(())
}

/// Run a named procedure with its `{key: value}` config, returning `(node, value)`
/// per node (component ids / labels surfaced as their `f64` node-id number). `None`
/// for an unknown name (the parser normally rejects that first). Reuses the
/// deterministic algorithms above.
#[must_use]
pub fn run_procedure(
    store: &Store,
    name: &str,
    config: &[(String, Value)],
) -> Option<Vec<(u32, Value)>> {
    // neighbor_aggregate produces a per-node feature VECTOR (a Value::List) and may
    // reject its config; a config error surfaces as `None` (CALL reports the failed
    // procedure). Every other procedure is scalar and is wrapped into Value::Num below.
    if name == "neighbor_aggregate" {
        return neighbor_aggregate(store, config).ok();
    }
    let str_of = |k: &str| {
        config.iter().find(|(ck, _)| ck == k).and_then(|(_, v)| {
            if let Value::Str(s) = v {
                Some(s.as_ref())
            } else {
                None
            }
        })
    };
    let num_of = |k: &str| {
        config.iter().find(|(ck, _)| ck == k).and_then(|(_, v)| {
            if let Value::Num(n) = v {
                Some(*n)
            } else {
                None
            }
        })
    };
    let dir = || match str_of("direction") {
        Some("in") => Dir::In,
        Some("both") => Dir::Both,
        _ => Dir::Out,
    };
    // Every remaining procedure is scalar `(node, f64)`; wrap into `(node, Value::Num)`.
    let numeric: Vec<(u32, f64)> = match name {
        "degree" => degree(store, dir(), str_of("edgeType")),
        "connected_components" => weakly_connected_components(store, str_of("edgeType"))
            .into_iter()
            .map(|(v, c)| (v, f64::from(c)))
            .collect(),
        "label_propagation" => {
            let iters = num_of("iterations").map_or(DEFAULT_LABEL_ITERATIONS, |n| n as u32);
            label_propagation(store, str_of("edgeType"), iters)
                .into_iter()
                .map(|(v, l)| (v, f64::from(l)))
                .collect()
        }
        "pagerank" => {
            let d = num_of("dampingFactor").unwrap_or(DEFAULT_DAMPING);
            let iters = num_of("iterations").map_or(DEFAULT_PAGERANK_ITERATIONS, |n| n as u32);
            pagerank(store, str_of("edgeType"), d, iters)
        }
        "closeness" => closeness(store, str_of("edgeType"), str_of("weightProperty")),
        "strongly_connected_components" => strongly_connected_components(store, str_of("edgeType"))
            .into_iter()
            .map(|(v, c)| (v, f64::from(c)))
            .collect(),
        "on_cycle" => on_cycle(store, str_of("edgeType")),
        "betweenness" => betweenness(store, str_of("edgeType"), str_of("weightProperty")),
        "shortest_path" => {
            if str_of("algorithm") == Some("astar") {
                // Goal-directed A*: source→target only, guided by heuristicProperty.
                astar_search(
                    store,
                    str_of("source"),
                    str_of("target"),
                    dir(),
                    str_of("edgeType"),
                    str_of("weightProperty"),
                    str_of("heuristicProperty"),
                )
            } else {
                shortest_path(
                    store,
                    str_of("source"),
                    dir(),
                    str_of("edgeType"),
                    str_of("weightProperty"),
                    str_of("target"),
                )
            }
        }
        "personalized_pagerank" => {
            let d = num_of("dampingFactor").unwrap_or(DEFAULT_DAMPING);
            let iters = num_of("iterations").map_or(DEFAULT_PAGERANK_ITERATIONS, |n| n as u32);
            // `sourceNodes` is a list of external-id strings (non-string items ignored).
            let seeds: Vec<String> = config
                .iter()
                .find(|(ck, _)| ck == "sourceNodes")
                .and_then(|(_, v)| {
                    if let Value::List(items) = v {
                        Some(
                            items
                                .iter()
                                .filter_map(|it| match it {
                                    Value::Str(s) => Some(s.to_string()),
                                    _ => None,
                                })
                                .collect(),
                        )
                    } else {
                        None
                    }
                })
                .unwrap_or_default();
            personalized_pagerank(store, str_of("edgeType"), &seeds, d, iters)
        }
        "peer_pressure" => {
            let iters = num_of("iterations").map_or(DEFAULT_PEER_PRESSURE_ITERATIONS, |n| n as u32);
            peer_pressure(store, str_of("edgeType"), iters)
                .into_iter()
                .map(|(v, c)| (v, f64::from(c)))
                .collect()
        }
        _ => return None,
    };
    // `on_cycle` yields a BOOLEAN flag (core surfaces `onCycle` as Bool, not a 0/1
    // number); every other scalar procedure yields a number.
    let to_value: fn(f64) -> Value = if name == "on_cycle" {
        |x| Value::Bool(x != 0.0)
    } else {
        Value::Num
    };
    Some(numeric.into_iter().map(|(v, x)| (v, to_value(x))).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Builder;

    /// A directed triangle a→b→c→a (ids 0,1,2) plus an isolated node d (id 3).
    fn triangle_plus_isolated() -> Store {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[]);
        let bb = b.node(&["N"], &[]);
        let c = b.node(&["N"], &[]);
        let _d = b.node(&["N"], &[]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        b.edge(c, a, "R");
        b.build()
    }

    #[test]
    fn degree_out_in_both() {
        let st = triangle_plus_isolated();
        // Each triangle node: 1 out, 1 in, 2 both; the isolated node: 0.
        assert_eq!(
            degree(&st, Dir::Out, None),
            vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0)]
        );
        assert_eq!(
            degree(&st, Dir::In, None),
            vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0)]
        );
        assert_eq!(
            degree(&st, Dir::Both, None),
            vec![(0, 2.0), (1, 2.0), (2, 2.0), (3, 0.0)]
        );
        // An unknown edge type → all zero.
        assert_eq!(
            degree(&st, Dir::Out, Some("NOPE")),
            vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
        );
    }

    #[test]
    fn closeness_reciprocal_of_summed_distances() {
        let st = triangle_plus_isolated();
        // Directed OUT on a→b→c→a: each triangle node reaches the other two at hop
        // distances 1 and 2, so Σ = 3 and closeness = 1/3. The isolated node reaches
        // nothing → sum 0 → closeness 0.
        assert_eq!(
            closeness(&st, None, None),
            vec![(0, 1.0 / 3.0), (1, 1.0 / 3.0), (2, 1.0 / 3.0), (3, 0.0)]
        );
        // A named-but-unknown edge type reaches only each source → every closeness 0.
        assert_eq!(
            closeness(&st, Some("NOPE"), None),
            vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
        );
    }

    #[test]
    fn scc_partitions_cycles_and_singletons() {
        let st = triangle_plus_isolated();
        // The directed triangle {0,1,2} is one SCC (rep = min member 0); the isolated
        // node {3} is its own singleton (rep 3).
        assert_eq!(
            strongly_connected_components(&st, None),
            vec![(0, 0), (1, 0), (2, 0), (3, 3)]
        );
        // A DAG (drop the closing c→a edge) has no non-trivial cycle → all singletons.
        let mut b = Builder::default();
        let a = b.node(&["N"], &[]);
        let bb = b.node(&["N"], &[]);
        let c = b.node(&["N"], &[]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        let dag = b.build();
        assert_eq!(
            strongly_connected_components(&dag, None),
            vec![(0, 0), (1, 1), (2, 2)]
        );
        // A named-but-unknown edge type → every vertex is its own singleton.
        assert_eq!(
            strongly_connected_components(&st, Some("NOPE")),
            vec![(0, 0), (1, 1), (2, 2), (3, 3)]
        );
    }

    #[test]
    fn on_cycle_flags_cycle_members_and_self_loops() {
        let st = triangle_plus_isolated();
        // Triangle members are on a cycle (1.0); the isolated node is not (0.0).
        assert_eq!(
            on_cycle(&st, None),
            vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0)]
        );
        // A lone self-loop is a 1-cycle: node 1 loops on itself, node 0 does not.
        let mut b = Builder::default();
        let _a = b.node(&["N"], &[]);
        let bb = b.node(&["N"], &[]);
        b.edge(bb, bb, "R");
        let loops = b.build();
        assert_eq!(on_cycle(&loops, None), vec![(0, 0.0), (1, 1.0)]);
        // A named-but-unknown edge type → nothing is on a cycle.
        assert_eq!(
            on_cycle(&st, Some("NOPE")),
            vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
        );
    }

    #[test]
    fn betweenness_directed_triangle_and_diamond() {
        // Directed triangle 0→1→2→0: each vertex is the sole intermediary of exactly
        // one 2-hop shortest path, so every betweenness is 1.0; the isolated node 0.
        let st = triangle_plus_isolated();
        assert_eq!(
            betweenness(&st, None, None),
            vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0)]
        );
        // Diamond 0→1, 0→2, 1→3, 2→3: from 0 there are TWO shortest paths to 3, so
        // each middle vertex carries HALF the dependency — exercises sigma division.
        let mut b = Builder::default();
        let a = b.node(&["N"], &[]);
        let l = b.node(&["N"], &[]);
        let r = b.node(&["N"], &[]);
        let d = b.node(&["N"], &[]);
        b.edge(a, l, "R");
        b.edge(a, r, "R");
        b.edge(l, d, "R");
        b.edge(r, d, "R");
        let diamond = b.build();
        assert_eq!(
            betweenness(&diamond, None, None),
            vec![(0, 0.0), (1, 0.5), (2, 0.5), (3, 0.0)]
        );
        // A named-but-unknown edge type → no paths → every vertex 0.0.
        assert_eq!(
            betweenness(&st, Some("NOPE"), None),
            vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
        );
    }

    #[test]
    fn personalized_pagerank_restarts_at_the_seed() {
        let st = triangle_plus_isolated();
        let ppr = |seeds: &[&str]| {
            let owned: Vec<String> = seeds.iter().map(|s| (*s).to_string()).collect();
            personalized_pagerank(
                &st,
                None,
                &owned,
                DEFAULT_DAMPING,
                DEFAULT_PAGERANK_ITERATIONS,
            )
        };
        // Seeding node 0 concentrates mass at 0, tapering around the directed cycle;
        // the unreachable isolated node gets none. (Exact values pin the f64 result.)
        let seed0 = ppr(&["0"]);
        assert_eq!(
            seed0,
            vec![
                (0, 0.375_920_077_192_677_5),
                (1, 0.319_532_065_613_775_9),
                (2, 0.304_547_857_193_546_66),
                (3, 0.0),
            ]
        );
        // Deterministic.
        assert_eq!(ppr(&["0"]), seed0);
        // Mass is conserved (a proper distribution over the reachable component).
        let mass: f64 = seed0.iter().map(|&(_, x)| x).sum();
        assert!((mass - 1.0).abs() < 1e-12, "mass {mass} should be ~1");
        // The seed is the strict maximum.
        assert!(seed0[0].1 > seed0[1].1 && seed0[0].1 > seed0[2].1);
        // Seeding node 1 is the rotational image (triangle symmetry).
        let seed1 = ppr(&["1"]);
        assert_eq!(seed1[1].1, seed0[0].1);
        assert_eq!(seed1[2].1, seed0[1].1);
        // With no resolvable seed it degenerates to global PageRank — so an unknown
        // id equals the empty-seed run, and there the isolated node gets teleport mass.
        let none = ppr(&[]);
        assert_eq!(ppr(&["999"]), none);
        assert!(none[3].1 > 0.0);
    }

    #[test]
    fn neighbor_aggregate_gcn_normalization() {
        // 0→1, 0→2 (unweighted); features 1=[2], 2=[4]. Degrees under the OUT filter:
        // deg[0]=2 (two contributors), deg[1]=deg[2]=1 (no out-neighbours, floored to
        // 1). Each contributor's GCN factor is 1/sqrt(deg_0·deg_nbr) = 1/sqrt(2), so
        // the GCN sum at 0 is 2/sqrt(2) + 4/sqrt(2) — folded in that order (the exact
        // f64 lenke-core produces, NOT a re-derived 3·sqrt(2), which rounds one ULP
        // off in the last place).
        let mut b = Builder::default();
        let f = |x: f64| Value::List(vec![Value::Num(x)]);
        b.node(&["N"], &[]);
        b.node(&["N"], &[("h", f(2.0))]);
        b.node(&["N"], &[("h", f(4.0))]);
        let mut st = b.build();
        st.add_edge(0, 1, "R");
        st.add_edge(0, 2, "R");

        let cfg = |op: &str, gcn: bool| {
            let mut c = vec![
                ("feature".to_string(), Value::Str("h".into())),
                ("op".to_string(), Value::Str(op.into())),
                ("direction".to_string(), Value::Str("out".into())),
            ];
            if gcn {
                c.push(("norm".to_string(), Value::Str("gcn".into())));
            }
            c
        };
        let node0 = |op: &str, gcn: bool| {
            format!("{:?}", neighbor_aggregate(&st, &cfg(op, gcn)).unwrap()[0].1)
        };
        // GCN sum: 2·(1/√2) + 4·(1/√2), in fold order.
        assert_eq!(node0("sum", true), "List([Num(4.242640687119285)])");
        // Un-normalized sum is just 2 + 4 = 6.
        assert_eq!(node0("sum", false), "List([Num(6.0)])");
        // A GCN norm is meaningless for max/min → Err; a bad norm value → Err.
        assert!(neighbor_aggregate(&st, &cfg("max", true)).is_err());
        assert!(neighbor_aggregate(
            &st,
            &[
                ("feature".into(), Value::Str("h".into())),
                ("norm".into(), Value::Str("l2".into())),
            ]
        )
        .is_err());
    }

    #[test]
    fn neighbor_aggregate_weighted_scales_by_edge_weight() {
        // 0→1 (w=1, feature [2]), 0→2 (w=3, feature [4]). Weighted sum at 0 is
        // 1·2 + 3·4 = 14; weighted mean divides by the WEIGHT sum: 14/(1+3) = 3.5 —
        // where the unweighted mean is (2+4)/2 = 3.
        let mut b = Builder::default();
        let f = |x: f64| Value::List(vec![Value::Num(x)]);
        b.node(&["N"], &[]);
        b.node(&["N"], &[("h", f(2.0))]);
        b.node(&["N"], &[("h", f(4.0))]);
        let mut st = b.build();
        let e0 = st.add_edge(0, 1, "R");
        st.set_edge_prop(e0, "w", Value::Num(1.0));
        let e1 = st.add_edge(0, 2, "R");
        st.set_edge_prop(e1, "w", Value::Num(3.0));

        let cfg = |op: &str, weighted: bool| {
            let mut c = vec![
                ("feature".to_string(), Value::Str("h".into())),
                ("op".to_string(), Value::Str(op.into())),
                ("direction".to_string(), Value::Str("out".into())),
            ];
            if weighted {
                c.push(("weightProperty".to_string(), Value::Str("w".into())));
            }
            c
        };
        let node0 = |op: &str, weighted: bool| {
            format!(
                "{:?}",
                neighbor_aggregate(&st, &cfg(op, weighted)).unwrap()[0].1
            )
        };
        assert_eq!(node0("sum", true), "List([Num(14.0)])");
        assert_eq!(node0("mean", true), "List([Num(3.5)])");
        assert_eq!(node0("mean", false), "List([Num(3.0)])");
        // A weight is meaningless for the scale-independent max/min → Err.
        assert!(neighbor_aggregate(&st, &cfg("max", true)).is_err());
        assert!(neighbor_aggregate(&st, &cfg("min", true)).is_err());
    }

    #[test]
    fn neighbor_aggregate_folds_feature_vectors() {
        // a(0)=[1,2], b(1)=[3,4], c(2)=[5,6]; edges a→b, a→c. OUT-aggregation at a
        // folds b and c; b and c have no out-neighbour → the zero vector.
        let mut b = Builder::default();
        let vec = |xs: &[f64]| Value::List(xs.iter().map(|&x| Value::Num(x)).collect());
        let a = b.node(&["N"], &[("h", vec(&[1.0, 2.0]))]);
        let bb = b.node(&["N"], &[("h", vec(&[3.0, 4.0]))]);
        let c = b.node(&["N"], &[("h", vec(&[5.0, 6.0]))]);
        b.edge(a, bb, "R");
        b.edge(a, c, "R");
        let st = b.build();

        let run = |op: &str, extra: &[(&str, Value)]| {
            let mut cfg: Vec<(String, Value)> = vec![
                ("feature".into(), Value::Str("h".into())),
                ("op".into(), Value::Str(op.into())),
                ("direction".into(), Value::Str("out".into())),
            ];
            cfg.extend(extra.iter().map(|(k, v)| ((*k).to_string(), v.clone())));
            neighbor_aggregate(&st, &cfg).unwrap()
        };
        let node0 = |op: &str, extra: &[(&str, Value)]| match &run(op, extra)[0].1 {
            Value::List(xs) => xs
                .iter()
                .map(|v| match v {
                    Value::Num(n) => *n,
                    _ => panic!("non-numeric"),
                })
                .collect::<Vec<f64>>(),
            _ => panic!("not a list"),
        };
        assert_eq!(node0("sum", &[]), vec![8.0, 10.0]); // 3+5, 4+6
        assert_eq!(node0("mean", &[]), vec![4.0, 5.0]); // /2 contributors
        assert_eq!(node0("max", &[]), vec![5.0, 6.0]);
        assert_eq!(node0("min", &[]), vec![3.0, 4.0]);
        // includeSelf folds a's own [1,2] into the sum → [9,12].
        assert_eq!(
            node0("sum", &[("includeSelf", Value::Bool(true))]),
            vec![9.0, 12.0]
        );
        // b(1) has no out-neighbour → the zero vector (mean does not divide by 0).
        assert_eq!(
            format!("{:?}", run("mean", &[])[1].1),
            "List([Num(0.0), Num(0.0)])"
        );
        // Config errors surface as Err.
        assert!(neighbor_aggregate(&st, &[]).is_err()); // missing feature
        assert!(neighbor_aggregate(
            &st,
            &[
                ("feature".into(), Value::Str("h".into())),
                ("op".into(), Value::Str("median".into())),
            ]
        )
        .is_err()); // bad op
    }

    #[test]
    fn peer_pressure_adopts_max_energy_cluster() {
        // Sink: 1→0, 2→0, 3→0. Node 0 sees equal vote energy from clusters 1, 2, 3
        // (each source has out-degree 1 → vote 1.0), so the tie goes to the smallest
        // external id, "1"; the sources have no in-edge and keep their own cluster.
        let mut b = Builder::default();
        let a = b.node(&["N"], &[]);
        let x = b.node(&["N"], &[]);
        let y = b.node(&["N"], &[]);
        let z = b.node(&["N"], &[]);
        b.edge(x, a, "R");
        b.edge(y, a, "R");
        b.edge(z, a, "R");
        let sink = b.build();
        assert_eq!(
            peer_pressure(&sink, None, DEFAULT_PEER_PRESSURE_ITERATIONS),
            vec![(0, 1), (1, 1), (2, 2), (3, 3)]
        );
        // A named-but-unknown edge type → every vertex its own cluster.
        assert_eq!(
            peer_pressure(&sink, Some("NOPE"), DEFAULT_PEER_PRESSURE_ITERATIONS),
            vec![(0, 0), (1, 1), (2, 2), (3, 3)]
        );
    }

    #[test]
    fn closeness_weighted_uses_dijkstra_distances() {
        // 0→1 (w=10), 0→2 (1), 2→1 (1). Weighted closeness of 0 sums the Dijkstra
        // distances (0 + 2 + 1 = 3 → 1/3), differing from the unweighted hop sum
        // (0 + 1 + 1 = 2 → 1/2). Node 2 reaches only 1 at cost 1 → 1/1; node 1 → 0.
        let mut b = Builder::default();
        b.node(&["N"], &[]);
        b.node(&["N"], &[]);
        b.node(&["N"], &[]);
        let mut st = b.build();
        let e0 = st.add_edge(0, 1, "R");
        st.set_edge_prop(e0, "w", Value::Num(10.0));
        let e1 = st.add_edge(0, 2, "R");
        st.set_edge_prop(e1, "w", Value::Num(1.0));
        let e2 = st.add_edge(2, 1, "R");
        st.set_edge_prop(e2, "w", Value::Num(1.0));

        assert_eq!(
            closeness(&st, None, Some("w")),
            vec![(0, 1.0 / 3.0), (1, 0.0), (2, 1.0)]
        );
        assert_eq!(
            closeness(&st, None, None),
            vec![(0, 1.0 / 2.0), (1, 0.0), (2, 1.0)]
        );
    }

    #[test]
    fn betweenness_weighted_reroutes_dependency() {
        // Diamond 0→1, 0→2, 1→3, 2→3 with the 2→3 branch heavy (w=5): the unique
        // weighted shortest 0→3 goes via 1, so node 1 carries the FULL dependency
        // (1.0) and node 2 none — where the UNWEIGHTED graph splits it 0.5/0.5.
        let mut b = Builder::default();
        for _ in 0..4 {
            b.node(&["N"], &[]);
        }
        let mut st = b.build();
        let e0 = st.add_edge(0, 1, "R");
        st.set_edge_prop(e0, "w", Value::Num(1.0));
        let e1 = st.add_edge(0, 2, "R");
        st.set_edge_prop(e1, "w", Value::Num(1.0));
        let e2 = st.add_edge(1, 3, "R");
        st.set_edge_prop(e2, "w", Value::Num(1.0));
        let e3 = st.add_edge(2, 3, "R");
        st.set_edge_prop(e3, "w", Value::Num(5.0));

        assert_eq!(
            betweenness(&st, None, Some("w")),
            vec![(0, 0.0), (1, 1.0), (2, 0.0), (3, 0.0)]
        );
        assert_eq!(
            betweenness(&st, None, None),
            vec![(0, 0.0), (1, 0.5), (2, 0.5), (3, 0.0)]
        );
    }

    #[test]
    fn astar_matches_dijkstra_distance() {
        // 0→1 (w=10), 0→2 (1), 2→1 (1): the shortest 0→1 is the light detour (2).
        let mut b = Builder::default();
        b.node(&["N"], &[("hdist", Value::Num(1.0))]); // admissible heuristic to node 1
        b.node(&["N"], &[("hdist", Value::Num(0.0))]);
        b.node(&["N"], &[("hdist", Value::Num(0.5))]);
        let mut st = b.build();
        let e0 = st.add_edge(0, 1, "R");
        st.set_edge_prop(e0, "w", Value::Num(10.0));
        let e1 = st.add_edge(0, 2, "R");
        st.set_edge_prop(e1, "w", Value::Num(1.0));
        let e2 = st.add_edge(2, 1, "R");
        st.set_edge_prop(e2, "w", Value::Num(1.0));

        // A* returns the exact shortest distance — the same as target-restricted
        // weighted Dijkstra — with or without a (admissible) heuristic.
        let dij = shortest_path(&st, Some("0"), Dir::Out, None, Some("w"), Some("1"));
        assert_eq!(dij, vec![(1, 2.0)]);
        assert_eq!(
            astar_search(&st, Some("0"), Some("1"), Dir::Out, None, Some("w"), None),
            dij
        );
        assert_eq!(
            astar_search(
                &st,
                Some("0"),
                Some("1"),
                Dir::Out,
                None,
                Some("w"),
                Some("hdist")
            ),
            dij
        );
        // Unweighted A* is the hop distance (direct edge to 1).
        assert_eq!(
            astar_search(&st, Some("0"), Some("1"), Dir::Out, None, None, None),
            vec![(1, 1.0)]
        );
        // A missing/unknown source or target → empty.
        assert!(astar_search(&st, Some("0"), Some("999"), Dir::Out, None, None, None).is_empty());
        assert!(astar_search(&st, None, Some("1"), Dir::Out, None, None, None).is_empty());
    }

    #[test]
    fn shortest_path_weighted_dijkstra() {
        // 0→1 (weight 10), 0→2 (1), 2→1 (1). The weighted shortest 0→1 is the light
        // 2-hop detour (1+1=2), NOT the direct heavy edge — so it differs from the
        // unweighted hop distance (1).
        let mut b = Builder::default();
        b.node(&["N"], &[]);
        b.node(&["N"], &[]);
        b.node(&["N"], &[]);
        let mut st = b.build();
        let e0 = st.add_edge(0, 1, "R");
        st.set_edge_prop(e0, "w", Value::Num(10.0));
        let e1 = st.add_edge(0, 2, "R");
        st.set_edge_prop(e1, "w", Value::Num(1.0));
        let e2 = st.add_edge(2, 1, "R");
        st.set_edge_prop(e2, "w", Value::Num(1.0));

        assert_eq!(
            shortest_path(&st, Some("0"), Dir::Out, None, Some("w"), None),
            vec![(0, 0.0), (1, 2.0), (2, 1.0)]
        );
        // Without the weight it is the plain hop distance (direct edge to 1).
        assert_eq!(
            shortest_path(&st, Some("0"), Dir::Out, None, None, None),
            vec![(0, 0.0), (1, 1.0), (2, 1.0)]
        );
        // A negative weight makes Dijkstra unsound → the empty result (core errs).
        let mut b2 = Builder::default();
        b2.node(&["N"], &[]);
        b2.node(&["N"], &[]);
        let mut st2 = b2.build();
        let en = st2.add_edge(0, 1, "R");
        st2.set_edge_prop(en, "w", Value::Num(-1.0));
        assert!(shortest_path(&st2, Some("0"), Dir::Out, None, Some("w"), None).is_empty());
    }

    #[test]
    fn shortest_path_bfs_layers_from_source() {
        let st = triangle_plus_isolated();
        // OUT from a(ext "0") on 0→1→2→0: 0@0, 1@1, 2@2; the isolated node is
        // unreachable, so it is absent from the result.
        assert_eq!(
            shortest_path(&st, Some("0"), Dir::Out, None, None, None),
            vec![(0, 0.0), (1, 1.0), (2, 2.0)]
        );
        // IN from a walks the cycle backwards: 0@0, then c(2)@1, then b(1)@2.
        assert_eq!(
            shortest_path(&st, Some("0"), Dir::In, None, None, None),
            vec![(0, 0.0), (1, 2.0), (2, 1.0)]
        );
        // Unknown source → nothing.
        assert!(shortest_path(&st, Some("999"), Dir::Out, None, None, None).is_empty());
        assert!(shortest_path(&st, None, Dir::Out, None, None, None).is_empty());
        // A named-but-unknown edge type reaches only the source.
        assert_eq!(
            shortest_path(&st, Some("0"), Dir::Out, Some("NOPE"), None, None),
            vec![(0, 0.0)]
        );
        // A `target` restricts the result to just that vertex's distance.
        assert_eq!(
            shortest_path(&st, Some("0"), Dir::Out, None, None, Some("2")),
            vec![(2, 2.0)]
        );
        // An unreachable target (isolated node 3) or an unknown id → nothing.
        assert!(shortest_path(&st, Some("0"), Dir::Out, None, None, Some("3")).is_empty());
        assert!(shortest_path(&st, Some("0"), Dir::Out, None, None, Some("999")).is_empty());
    }

    #[test]
    fn wcc_labels_by_smallest_member() {
        let st = triangle_plus_isolated();
        // Triangle {0,1,2} → component 0 (min); isolated 3 → component 3.
        assert_eq!(
            weakly_connected_components(&st, None),
            vec![(0, 0), (1, 0), (2, 0), (3, 3)]
        );
        // No edges considered (unknown type) → every node its own component.
        assert_eq!(
            weakly_connected_components(&st, Some("NOPE")),
            vec![(0, 0), (1, 1), (2, 2), (3, 3)]
        );
    }

    #[test]
    fn bfs_distances_from_a_source() {
        let st = triangle_plus_isolated();
        // Out from 0: 0→b(1)→c(2); d unreachable.
        assert_eq!(
            bfs_distances(&st, 0, Dir::Out, None),
            vec![(0, 0), (1, 1), (2, 2)]
        );
        // Both from 0: c is one hop away via the incoming c→a edge.
        assert_eq!(
            bfs_distances(&st, 0, Dir::Both, None),
            vec![(0, 0), (1, 1), (2, 1)]
        );
        // A deleted / out-of-range source reaches nothing.
        assert!(bfs_distances(&st, 9, Dir::Out, None).is_empty());
    }

    #[test]
    fn label_propagation_converges_a_triangle() {
        let st = triangle_plus_isolated();
        // The undirected triangle collapses to one label (its smallest member id,
        // 0); the isolated node keeps its own label (3).
        assert_eq!(
            label_propagation(&st, None, DEFAULT_LABEL_ITERATIONS),
            vec![(0, 0), (1, 0), (2, 0), (3, 3)]
        );
        // No edges (unknown type) → every node keeps its own label.
        assert_eq!(
            label_propagation(&st, Some("NOPE"), DEFAULT_LABEL_ITERATIONS),
            vec![(0, 0), (1, 1), (2, 2), (3, 3)]
        );
    }

    #[test]
    fn label_propagation_parallel_round_matches_sequential() {
        // A graph past the parallel threshold (>= 8192 nodes) with deterministic edges.
        let mut b = Builder::default();
        let nn = 9000u32;
        for _ in 0..nn {
            b.node(&["N"], &[]);
        }
        for i in 0..nn {
            b.edge(i, (i.wrapping_mul(7).wrapping_add(1)) % nn, "R");
            b.edge(i, (i + 1) % nn, "R");
        }
        let st = b.build();
        let n = st.node_count();
        let live = st.all_nodes();
        let csr = Csr::build(&st, Dir::Both, want_etype(&st, Some("R")).unwrap(), n);
        // The 8-thread round must equal the single-thread round, round after round.
        let mut seq: Vec<u32> = (0..n as u32).collect();
        let mut par = seq.clone();
        for _ in 0..3 {
            seq = lp_round(&live, &csr, &seq, n, 1);
            par = lp_round(&live, &csr, &par, n, 8);
            assert_eq!(seq, par);
        }
        // And the public entry (which picks the parallel path here) agrees with a forced
        // single-thread run to convergence.
        let got = label_propagation(&st, Some("R"), DEFAULT_LABEL_ITERATIONS);
        let mut want: Vec<u32> = (0..n as u32).collect();
        for _ in 0..DEFAULT_LABEL_ITERATIONS {
            let nxt = lp_round(&live, &csr, &want, n, 1);
            if nxt == want {
                break;
            }
            want = nxt;
        }
        let want_pairs: Vec<(u32, u32)> = live.iter().map(|&v| (v, want[v as usize])).collect();
        assert_eq!(got, want_pairs);
    }

    #[test]
    fn pagerank_two_cycle_is_uniform_and_sums_to_one() {
        // a ↔ b (a→b, b→a): symmetric, no dangling → each rank is exactly 0.5.
        let mut b = Builder::default();
        let a = b.node(&["N"], &[]);
        let bb = b.node(&["N"], &[]);
        b.edge(a, bb, "R");
        b.edge(bb, a, "R");
        let st = b.build();
        let pr = pagerank(&st, None, DEFAULT_DAMPING, DEFAULT_PAGERANK_ITERATIONS);
        assert!((pr[0].1 - 0.5).abs() < 1e-12);
        assert!((pr[1].1 - 0.5).abs() < 1e-12);
        let total: f64 = pr.iter().map(|(_, r)| r).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pagerank_ranks_higher_in_degree_and_is_reproducible() {
        // 0→2, 1→2, 0→1: node 2 (in-degree 2) outranks node 1 (in-degree 1), which
        // outranks the source-only node 0.
        let mut b = Builder::default();
        let n0 = b.node(&["N"], &[]);
        let n1 = b.node(&["N"], &[]);
        let n2 = b.node(&["N"], &[]);
        b.edge(n0, n2, "R");
        b.edge(n1, n2, "R");
        b.edge(n0, n1, "R");
        let st = b.build();
        let pr = pagerank(&st, None, DEFAULT_DAMPING, DEFAULT_PAGERANK_ITERATIONS);
        assert!(pr[2].1 > pr[1].1, "hub should outrank {pr:?}");
        assert!(
            pr[1].1 > pr[0].1,
            "middle should outrank the source-only {pr:?}"
        );
        let total: f64 = pr.iter().map(|(_, r)| r).sum();
        assert!((total - 1.0).abs() < 1e-9);
        // Deterministic: the same input gives a bit-identical result.
        assert_eq!(
            pr,
            pagerank(&st, None, DEFAULT_DAMPING, DEFAULT_PAGERANK_ITERATIONS)
        );
    }
}
