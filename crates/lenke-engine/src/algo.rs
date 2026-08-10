//! In-engine graph algorithms over [`crate::store::Store`]. Each returns a
//! per-node result keyed by the node's dense id, in ascending-id order, so the
//! output is DETERMINISTIC and reproducible (the same rule lenke-core follows so
//! the two engines agree). Deleted (tombstoned) nodes are skipped.
//!
//! Five algorithms: the non-iterative trio — degree, weakly connected components
//! (union-by-min), and BFS distances — plus the iterative pair, PageRank and label
//! propagation, whose f64 summation / tiebreak order is pinned (node-id and
//! in-adjacency order) so their results are reproducible and match lenke-core.

use crate::ir::Dir;
use crate::store::Store;
use crate::value::Value;
use std::collections::HashMap;

/// PageRank defaults, matching lenke-core.
pub const DEFAULT_DAMPING: f64 = 0.85;
pub const DEFAULT_PAGERANK_ITERATIONS: u32 = 20;
/// Label-propagation default round bound, matching lenke-core.
pub const DEFAULT_LABEL_ITERATIONS: u32 = 10;

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
            for_each_nbr(store, v, Dir::In, want, |u| {
                sum += pr[u as usize] / f64::from(outdeg[u as usize]);
            });
            next[v as usize] = base + damping * sum;
        }
        pr = next;
    }
    live.into_iter().map(|v| (v, pr[v as usize])).collect()
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
        for _ in 0..iterations {
            let mut next = labels.clone();
            for &v in &live {
                let mut counts: HashMap<u32, u32> = HashMap::new();
                for_each_nbr(store, v, Dir::Both, want, |u| {
                    *counts.entry(labels[u as usize]).or_insert(0) += 1;
                });
                // Most-frequent label; tie → smallest label id. No neighbours → keep.
                let mut best: Option<(u32, u32)> = None; // (label, count)
                for (&lbl, &c) in &counts {
                    let better = match best {
                        None => true,
                        Some((bl, bc)) => c > bc || (c == bc && lbl < bl),
                    };
                    if better {
                        best = Some((lbl, c));
                    }
                }
                if let Some((lbl, _)) = best {
                    next[v as usize] = lbl;
                }
            }
            if next == labels {
                break; // converged
            }
            labels = next;
        }
    }
    live.into_iter().map(|v| (v, labels[v as usize])).collect()
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
        "label_propagation" => "label",
        _ => return None,
    })
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
) -> Option<Vec<(u32, f64)>> {
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
    Some(match name {
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
        _ => return None,
    })
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
