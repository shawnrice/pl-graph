//! In-engine graph algorithms over [`crate::store::Store`]. Each returns a
//! per-node result keyed by the node's dense id, in ascending-id order, so the
//! output is DETERMINISTIC and reproducible (the same rule lenke-core follows so
//! the two engines agree). Deleted (tombstoned) nodes are skipped.
//!
//! This slice (I1a) is the deterministic, non-iterative trio — degree, weakly
//! connected components (union-by-min), and BFS distances. The iterative
//! algorithms (label propagation, PageRank) need a fixed summation/tiebreak order
//! for byte-identity and land in I1b.

use crate::ir::Dir;
use crate::store::Store;

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
}
